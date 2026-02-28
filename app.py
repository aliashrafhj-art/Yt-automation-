import os, subprocess, sys, json, time, threading, requests, uuid, traceback, re, base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, session, send_file, make_response

# Install deps if missing (for non-Railway envs)
try:
    import google.generativeai
except ImportError:
    os.system("pip install google-generativeai --break-system-packages --quiet")

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change-me-to-random-secret')
app.permanent_session_lifetime = 86400 * 30

# Railway sets PORT automatically
PORT = int(os.environ.get('PORT', 5000))

YOUTUBE_CLIENT_ID     = os.environ.get('YOUTUBE_CLIENT_ID', '')
YOUTUBE_CLIENT_SECRET = os.environ.get('YOUTUBE_CLIENT_SECRET', '')
DEFAULT_GEMINI_KEY    = os.environ.get('GEMINI_API_KEY', '')  # optional default

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# In-memory job store (per-process)
# Key: job_id, Value: job dict
jobs = {}

# ─── SCHEDULES (per-user, stored in session) ───
def get_user_schedules():
    return session.get('schedules', [])

def save_user_schedules(scheds):
    session['schedules'] = scheds
    session.modified = True

# ─── REDIRECT URI (auto-detect Railway URL) ───
def get_redirect_uri():
    # Railway provides RAILWAY_PUBLIC_DOMAIN
    railway_domain = os.environ.get('RAILWAY_PUBLIC_DOMAIN', '')
    if railway_domain:
        return f"https://{railway_domain}/oauth_callback"
    # Manual override
    override = os.environ.get('REDIRECT_URI', '')
    if override:
        return override
    # Fallback for local dev
    return 'http://localhost:5000/oauth_callback'

def get_client_config():
    redirect_uri = get_redirect_uri()
    return {
        "web": {
            "client_id": YOUTUBE_CLIENT_ID,
            "client_secret": YOUTUBE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri]
        }
    }

# ─── CREDENTIALS ───
def creds_from_session():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        token_data = session.get('yt_token')
        if not token_data:
            return None
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            session['yt_token'] = json.loads(creds.to_json())
            session.modified = True
        return creds if creds and creds.valid else None
    except:
        return None

def get_gemini_key_for_request():
    """Get Gemini key: from request body > session > env default"""
    # From request JSON body
    try:
        data = request.get_json(silent=True) or {}
        if data.get('gemini_key'):
            return data['gemini_key'].strip()
    except:
        pass
    # From session (user saved it)
    if session.get('gemini_key'):
        return session['gemini_key']
    # From environment (admin default)
    return DEFAULT_GEMINI_KEY

# ─── ROUTES: PAGES ───
@app.route('/')
def index():
    authenticated = creds_from_session() is not None
    has_gemini = bool(session.get('gemini_key') or DEFAULT_GEMINI_KEY)
    redirect_uri = get_redirect_uri()
    resp = make_response(render_template('index.html',
        authenticated=authenticated,
        has_gemini=has_gemini,
        redirect_uri=redirect_uri,
        client_id_set=bool(YOUTUBE_CLIENT_ID)
    ))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp

@app.route('/check')
def check():
    return jsonify({
        'ok': True,
        'authenticated': creds_from_session() is not None,
        'redirect_uri': get_redirect_uri(),
        'client_id_set': bool(YOUTUBE_CLIENT_ID),
        'gemini_env_set': bool(DEFAULT_GEMINI_KEY),
        'ffmpeg': bool(subprocess.run(['which','ffmpeg'], capture_output=True).stdout),
        'yt_dlp': bool(subprocess.run(['which','yt-dlp'], capture_output=True).stdout),
    })

# ─── ROUTES: AUTH ───
@app.route('/save_gemini_key', methods=['POST'])
def save_gemini_key():
    key = (request.json or {}).get('key', '').strip()
    if not key:
        return jsonify({'error': 'No key provided'}), 400
    session['gemini_key'] = key
    session.modified = True
    return jsonify({'status': 'saved'})

@app.route('/get_auth_url')
def get_auth_url():
    if not YOUTUBE_CLIENT_ID or not YOUTUBE_CLIENT_SECRET:
        return jsonify({'error': 'YOUTUBE_CLIENT_ID ও YOUTUBE_CLIENT_SECRET env variable সেট করা নেই।'}), 400
    try:
        from google_auth_oauthlib.flow import Flow
        redirect_uri = get_redirect_uri()
        flow = Flow.from_client_config(get_client_config(), scopes=SCOPES, redirect_uri=redirect_uri)
        auth_url, state = flow.authorization_url(prompt='consent', access_type='offline')
        session['oauth_state'] = state
        session.modified = True
        return jsonify({'url': auth_url, 'redirect_uri': redirect_uri})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/oauth_callback')
def oauth_callback():
    code = request.args.get('code')
    if request.args.get('error'):
        return redirect(f'/?auth_error={request.args.get("error")}')
    if not code:
        return redirect('/?auth_error=no_code')
    try:
        from google_auth_oauthlib.flow import Flow
        redirect_uri = get_redirect_uri()
        flow = Flow.from_client_config(get_client_config(), scopes=SCOPES, redirect_uri=redirect_uri)
        flow.fetch_token(code=code)
        creds = flow.credentials
        session['yt_token'] = json.loads(creds.to_json())
        session.permanent = True
        session.modified = True
        return redirect('/?auth_success=1')
    except Exception as e:
        return redirect(f'/?auth_error={str(e)[:120]}')

@app.route('/logout')
def logout():
    session.pop('yt_token', None)
    return redirect('/')

# ─── ROUTES: ANALYZE ───
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json or {}
    url = data.get('url', '').strip()
    num_clips = max(1, min(int(data.get('num_clips', 5)), 20))
    gemini_key = get_gemini_key_for_request()

    if not url:
        return jsonify({'error': 'URL দাও'}), 400
    if not gemini_key:
        return jsonify({'error': 'GEMINI_KEY_MISSING'}), 400

    job_id = f"analyze_{uuid.uuid4().hex[:10]}"
    jobs[job_id] = {
        'type': 'analyze', 'status': 'শুরু হচ্ছে...', 'progress': 0,
        'url': url, 'num_clips': num_clips,
        'transcript': None, 'clips': [], 'error': None,
        'owner': session.get('_id', 'anon')
    }
    threading.Thread(
        target=run_analyze,
        args=(job_id, url, num_clips, gemini_key),
        daemon=True
    ).start()
    return jsonify({'job_id': job_id})

def run_analyze(job_id, url, num_clips, gemini_key):
    j = jobs[job_id]
    try:
        # Extract video ID
        if 'v=' in url:
            video_id = url.split('v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        else:
            raise Exception('Invalid YouTube URL')

        # Video info
        j['status'] = 'Video info নিচ্ছি...'
        j['progress'] = 5
        info_r = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-download', url],
            capture_output=True, text=True, timeout=30
        )
        duration = 3600
        title = 'YouTube Video'
        if info_r.returncode == 0:
            try:
                info = json.loads(info_r.stdout)
                duration = info.get('duration', 3600)
                title = info.get('title', 'YouTube Video')
            except:
                pass
        j['title'] = title
        j['duration'] = duration

        # Heatmap
        j['status'] = 'Most replayed data খুঁজছি...'
        j['progress'] = 12
        heatmap_peaks = []
        try:
            r = requests.get(
                f'https://yt.lemnoslife.com/videos?part=mostReplayed&id={video_id}',
                timeout=10
            ).json()
            markers = r['items'][0]['mostReplayed']['markers']
            sorted_m = sorted(markers, key=lambda m: m['intensityScoreNormalized'], reverse=True)
            heatmap_peaks = [m['startMillis']/1000.0 for m in sorted_m[:15]]
        except:
            pass

        # Transcript: try subtitles first
        j['status'] = 'Subtitle খুঁজছি...'
        j['progress'] = 20
        transcript_text = None
        out_prefix = f'/tmp/{job_id}'

        subprocess.run([
            'yt-dlp', '--write-auto-sub', '--write-sub',
            '--sub-lang', 'en', '--skip-download',
            '--sub-format', 'vtt', '-o', out_prefix, url
        ], capture_output=True, text=True, timeout=60)

        vtt_file = next(
            (f'/tmp/{f}' for f in os.listdir('/tmp') if f.startswith(job_id) and f.endswith('.vtt')),
            None
        )

        if vtt_file:
            j['status'] = 'Subtitle parse করছি...'
            j['progress'] = 32
            with open(vtt_file) as f:
                segments = parse_vtt(f.read())
            transcript_text = ' '.join(s['text'] for s in segments)
            try:
                os.remove(vtt_file)
            except:
                pass
        else:
            # Download audio → Gemini transcribe
            j['status'] = 'Subtitle নেই — Audio নামাচ্ছি...'
            j['progress'] = 25
            audio_file = f'/tmp/{job_id}_audio.mp3'
            dl = subprocess.run([
                'yt-dlp', '-f', 'bestaudio',
                '--extract-audio', '--audio-format', 'mp3',
                '--audio-quality', '5', '-o', audio_file, url
            ], capture_output=True, text=True, timeout=600)

            if dl.returncode == 0 and os.path.exists(audio_file):
                j['status'] = 'Gemini দিয়ে transcript বানাচ্ছি...'
                j['progress'] = 45
                transcript_text = gemini_transcribe(audio_file, gemini_key)
                try:
                    os.remove(audio_file)
                except:
                    pass
            else:
                transcript_text = f'[Subtitle পাওয়া যায়নি। Title: {title}]'

        j['transcript'] = (transcript_text or '')[:8000]

        # Gemini: find best clips
        j['status'] = f'Gemini দিয়ে best {num_clips}টা moment খুঁজছি...'
        j['progress'] = 65
        clips = gemini_find_best_clips(
            transcript_text, heatmap_peaks, duration, num_clips, title, gemini_key
        )

        j['clips'] = clips
        j['status'] = 'Done'
        j['progress'] = 100

    except Exception as e:
        j['status'] = f'Error: {str(e)}'
        j['error'] = traceback.format_exc()
        j['progress'] = 0


def gemini_transcribe(audio_file, api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Compress if too large (>15MB)
        if os.path.getsize(audio_file) > 15 * 1024 * 1024:
            compressed = audio_file.replace('.mp3', '_c.mp3')
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_file,
                '-ab', '32k', '-ar', '16000', '-ac', '1', compressed
            ], capture_output=True, timeout=120)
            if os.path.exists(compressed):
                audio_file = compressed

        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()

        model = genai.GenerativeModel('gemini-1.5-flash')
        resp = model.generate_content([
            {'inline_data': {
                'mime_type': 'audio/mp3',
                'data': base64.b64encode(audio_bytes).decode()
            }},
            'Transcribe this audio with timestamps every ~30 seconds. Format: [MM:SS] text. Include all speech.'
        ])
        return resp.text
    except Exception as e:
        return f'[Gemini transcription error: {str(e)[:100]}]'


def gemini_find_best_clips(transcript, heatmap_peaks, duration, num_clips, title, api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        heatmap_str = ''
        if heatmap_peaks:
            heatmap_str = f'\nMost replayed peaks (sec): {[round(p) for p in heatmap_peaks[:10]]}'

        prompt = f"""Video: "{title}"
Duration: {int(duration//60)}m {int(duration%60)}s{heatmap_str}

Transcript:
{(transcript or '')[:5500]}

Find exactly {num_clips} best YouTube Shorts moments (45-60 seconds each).
Rules:
- No overlapping clips
- Pick viral-worthy, high-engagement moments
- Prioritize moments near heatmap peaks if available
- Each clip needs a strong hook in first 3 seconds

Return ONLY a JSON array:
[{{"start":120,"end":175,"title":"Catchy title","reason":"Why viral (1 line)","hook":"First 3 sec hook"}}]"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        resp = model.generate_content(prompt)
        text = re.sub(r'```(?:json)?', '', resp.text).strip().strip('`').strip()
        clips_raw = json.loads(text)

        validated = []
        used = []
        for c in clips_raw[:num_clips]:
            s = max(0.0, float(c.get('start', 0)))
            e = min(float(duration), float(c.get('end', s + 50)))
            if e - s < 10:
                e = s + 50
            if e > duration:
                s = max(0, duration - 55)
                e = duration

            overlap = any(not (e <= us or s >= ue) for us, ue in used)
            if not overlap:
                validated.append({
                    'start': round(s), 'end': round(e),
                    'title': str(c.get('title', f'Clip {len(validated)+1}')),
                    'reason': str(c.get('reason', '')),
                    'hook': str(c.get('hook', '')),
                    'duration': round(e - s)
                })
                used.append((s, e))
        return validated

    except Exception as e:
        # Fallback: evenly distributed
        clips = []
        interval = duration / (num_clips + 1)
        for i in range(num_clips):
            s = round(interval * (i + 1) - 25)
            e = s + 50
            if e > duration:
                e = duration; s = max(0, e - 50)
            clips.append({'start': s, 'end': e,
                          'title': f'Clip {i+1}', 'reason': f'Auto (Gemini error: {str(e)[:40]})',
                          'hook': '', 'duration': 50})
        return clips

# ─── ROUTES: GENERATE CLIPS ───
@app.route('/generate_all_clips', methods=['POST'])
def generate_all_clips():
    data = request.json or {}
    url = data.get('url', '').strip()
    clips_data = data.get('clips', [])
    if not url or not clips_data:
        return jsonify({'error': 'URL ও clips দাও'}), 400

    batch_id = uuid.uuid4().hex[:8]
    job_ids = []
    for clip in clips_data[:20]:
        jid = f"{batch_id}_{uuid.uuid4().hex[:5]}"
        jobs[jid] = {
            'type': 'clip', 'status': 'অপেক্ষায়...', 'progress': 0,
            'result_url': None, 'crop_ready': False, 'crop_file': None,
            'video_url': url,
            'title': clip.get('title', 'Short'),
            'caption': f"#shorts #viral\n\n{clip.get('reason','')}",
            'tags': 'shorts,viral', 'privacy': 'public',
            'clip_start': float(clip.get('start', 0)),
            'clip_end': float(clip.get('end', 50)),
            'clip_reason': clip.get('reason', ''),
            'clip_hook': clip.get('hook', ''),
        }
        job_ids.append(jid)

    threading.Thread(
        target=download_and_clip_all,
        args=(batch_id, url, job_ids, clips_data),
        daemon=True
    ).start()
    return jsonify({'job_ids': job_ids, 'batch_id': batch_id})

@app.route('/generate_single_clip', methods=['POST'])
def generate_single_clip():
    data = request.json or {}
    url = data.get('url', '').strip()
    start = float(data.get('start', 0))
    end = float(data.get('end', start + 50))
    clip_title = data.get('title', 'Short')

    jid = f"clip_{uuid.uuid4().hex[:8]}"
    jobs[jid] = {
        'type': 'clip', 'status': 'শুরু হচ্ছে...', 'progress': 0,
        'result_url': None, 'crop_ready': False, 'crop_file': None,
        'video_url': url, 'title': clip_title,
        'caption': '#shorts #viral', 'tags': 'shorts,viral', 'privacy': 'public',
        'clip_start': start, 'clip_end': end,
    }
    threading.Thread(target=process_single_clip, args=(jid, url, start, end), daemon=True).start()
    return jsonify({'job_id': jid})

def download_and_clip_all(batch_id, url, job_ids, clips_data):
    raw_file = f'/tmp/{batch_id}_raw.mp4'
    for jid in job_ids:
        jobs[jid]['status'] = 'Video নামাচ্ছি...'
        jobs[jid]['progress'] = 10
    try:
        dl = subprocess.run([
            'yt-dlp', '-f',
            'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4', '-o', raw_file, url
        ], capture_output=True, text=True, timeout=900)

        if dl.returncode != 0:
            raise Exception(f'Download failed: {dl.stderr[-200:]}')

        if not os.path.exists(raw_file):
            raw_file = next(
                (f'/tmp/{f}' for f in os.listdir('/tmp') if f.startswith(f'{batch_id}_raw')),
                None
            )
            if not raw_file:
                raise Exception('Downloaded file not found')

        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', raw_file
        ], capture_output=True, text=True)
        total_dur = float(probe.stdout.strip()) if probe.stdout.strip() else 99999

        for jid in job_ids:
            jobs[jid]['status'] = 'Clip কাটছি...'
            jobs[jid]['progress'] = 40

        threads = []
        for i, (jid, clip) in enumerate(zip(job_ids, clips_data)):
            t = threading.Thread(
                target=cut_clip,
                args=(jid, raw_file, float(clip.get('start', 0)),
                      float(clip.get('end', 50)), total_dur, i),
                daemon=True
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        try:
            os.remove(raw_file)
        except:
            pass

    except Exception as e:
        for jid in job_ids:
            jobs[jid]['status'] = f'Error: {str(e)}'
            jobs[jid]['progress'] = 0
        try:
            os.remove(raw_file)
        except:
            pass

def cut_clip(job_id, raw_file, start, end, total_dur, idx=0):
    try:
        start = max(0.0, min(start, total_dur - 10))
        end = min(end, total_dur)
        length = max(end - start, 10)
        jobs[job_id]['status'] = f'Clip {idx+1} বানাচ্ছি...'
        jobs[job_id]['progress'] = 50 + idx

        crop_file = f'/tmp/{job_id}_short.mp4'
        ff = subprocess.run([
            'ffmpeg', '-y',
            '-ss', str(start), '-t', str(length),
            '-i', raw_file,
            '-vf', 'crop=in_h*9/16:in_h,scale=1080:1920',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            crop_file
        ], capture_output=True, text=True, timeout=300)

        if ff.returncode != 0:
            raise Exception(f'FFmpeg: {ff.stderr[-150:]}')

        jobs[job_id].update({
            'status': 'Clip ready! ✅',
            'progress': 80,
            'crop_ready': True,
            'crop_file': crop_file
        })
    except Exception as e:
        jobs[job_id]['status'] = f'Error: {str(e)}'
        jobs[job_id]['progress'] = 0

def process_single_clip(job_id, url, start, end):
    raw_file = f'/tmp/{job_id}_raw.mp4'
    try:
        jobs[job_id]['status'] = 'Video নামাচ্ছি...'
        jobs[job_id]['progress'] = 15
        dl = subprocess.run([
            'yt-dlp', '-f',
            'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4', '-o', raw_file, url
        ], capture_output=True, text=True, timeout=900)

        if dl.returncode != 0:
            raise Exception(f'Download failed: {dl.stderr[-200:]}')

        if not os.path.exists(raw_file):
            raw_file = next(
                (f'/tmp/{f}' for f in os.listdir('/tmp') if f.startswith(f'{job_id}_raw')),
                None
            )

        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', raw_file
        ], capture_output=True, text=True)
        total_dur = float(probe.stdout.strip()) if probe.stdout.strip() else 99999

        cut_clip(job_id, raw_file, start, end, total_dur, 0)
        try:
            os.remove(raw_file)
        except:
            pass
    except Exception as e:
        jobs[job_id]['status'] = f'Error: {str(e)}'
        jobs[job_id]['progress'] = 0

# ─── ROUTES: STATUS / PREVIEW ───
@app.route('/status/<job_id>')
def status(job_id):
    return jsonify(jobs.get(job_id, {'status': 'Not found', 'progress': 0}))

@app.route('/status_batch', methods=['POST'])
def status_batch():
    ids = (request.json or {}).get('job_ids', [])
    return jsonify({jid: jobs.get(jid, {'status': 'Not found'}) for jid in ids})

@app.route('/preview/<job_id>')
def preview(job_id):
    crop_file = jobs.get(job_id, {}).get('crop_file')
    if not crop_file or not os.path.exists(crop_file):
        return 'File not found', 404
    return send_file(crop_file, mimetype='video/mp4')

# ─── ROUTES: UPLOAD ───
@app.route('/upload_now/<job_id>', methods=['POST'])
def upload_now(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    creds = creds_from_session()
    if not creds:
        try:
            from google_auth_oauthlib.flow import Flow
            redirect_uri = get_redirect_uri()
            flow = Flow.from_client_config(get_client_config(), scopes=SCOPES, redirect_uri=redirect_uri)
            auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
            return jsonify({'need_auth': True, 'auth_url': auth_url})
        except Exception as e:
            return jsonify({'need_auth': True, 'error': str(e)})

    data = request.json or {}
    jobs[job_id].update({
        'title': data.get('title', jobs[job_id].get('title', '')),
        'caption': data.get('caption', jobs[job_id].get('caption', '')),
        'tags': data.get('tags', jobs[job_id].get('tags', '')),
        'privacy': data.get('privacy', 'public')
    })
    threading.Thread(target=do_upload, args=(job_id, creds), daemon=True).start()
    return jsonify({'status': 'uploading'})

def do_upload(job_id, creds):
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        jobs[job_id]['status'] = 'YouTube এ upload হচ্ছে...'
        jobs[job_id]['progress'] = 85

        crop_file = jobs[job_id].get('crop_file')
        if not crop_file or not os.path.exists(crop_file):
            raise Exception('Clip file not found. Please regenerate.')

        title = (jobs[job_id].get('title') or 'YouTube Short #shorts')[:100]
        desc = jobs[job_id].get('caption') or '#shorts #viral'
        tags_raw = jobs[job_id].get('tags', 'shorts,viral')
        tags = [t.strip() for t in tags_raw.split(',') if t.strip()]
        privacy = jobs[job_id].get('privacy', 'public')

        youtube = build('youtube', 'v3', credentials=creds)
        body = {
            'snippet': {
                'title': title,
                'description': desc[:5000],
                'categoryId': '22',
                'tags': tags[:30]
            },
            'status': {
                'privacyStatus': privacy,
                'selfDeclaredMadeForKids': False
            }
        }
        media = MediaFileUpload(crop_file, chunksize=-1, resumable=True, mimetype='video/mp4')
        req = youtube.videos().insert(part='snippet,status', body=body, media_body=media)

        response = None
        while response is None:
            st, response = req.next_chunk()
            if st:
                pct = int(st.progress() * 100)
                jobs[job_id]['status'] = f'Upload হচ্ছে... {pct}%'
                jobs[job_id]['progress'] = 85 + int(pct * 0.14)

        jobs[job_id]['status'] = 'Done! ✅'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['result_url'] = f"https://youtube.com/shorts/{response['id']}"

        try:
            os.remove(crop_file)
        except:
            pass

    except Exception as e:
        jobs[job_id]['status'] = f'Upload Error: {str(e)}'
        print(traceback.format_exc())

# ─── ROUTES: SCHEDULE (per-user via session) ───
@app.route('/schedules')
def get_schedules():
    return jsonify(get_user_schedules())

@app.route('/schedule', methods=['POST'])
def add_schedule():
    data = request.json or {}
    s = {
        'id': uuid.uuid4().hex[:8],
        'job_id': data.get('job_id'),
        'title': data.get('title', ''),
        'caption': data.get('caption', ''),
        'tags': data.get('tags', ''),
        'privacy': data.get('privacy', 'public'),
        'scheduled_time': data.get('scheduled_time'),
        'status': 'pending',
        'created_at': datetime.now().isoformat()
    }
    scheds = get_user_schedules()
    scheds.append(s)
    save_user_schedules(scheds)
    return jsonify({'status': 'scheduled', 'schedule': s})

@app.route('/schedule/<sched_id>', methods=['DELETE'])
def delete_schedule(sched_id):
    scheds = [s for s in get_user_schedules() if s['id'] != sched_id]
    save_user_schedules(scheds)
    return jsonify({'status': 'deleted'})

# ─── VTT PARSER ───
def parse_vtt(content):
    segments = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        if '-->' in lines[i]:
            try:
                parts = lines[i].strip().split(' --> ')
                start = vtt_t(parts[0])
                end = vtt_t(parts[1].split(' ')[0])
                texts = []
                i += 1
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    t = re.sub(r'<[^>]+>', '', lines[i].strip())
                    if t:
                        texts.append(t)
                    i += 1
                if texts:
                    segments.append({'start': start, 'end': end, 'text': ' '.join(texts)})
                continue
            except:
                pass
        i += 1
    return segments

def vtt_t(t):
    t = t.strip().replace(',', '.')
    p = t.split(':')
    if len(p) == 3:
        return float(p[0])*3600 + float(p[1])*60 + float(p[2])
    if len(p) == 2:
        return float(p[0])*60 + float(p[1])
    return float(p[0])

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run(host='0.0.0.0', port=PORT, debug=False)

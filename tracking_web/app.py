"""单目标跟踪方法与智能分析系统 - Flask后端"""
import os, sys, cv2, time, json, base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import config
from wrappers.traditional_wrappers import TraditionalTrackerWrapper, TRADITIONAL_METHODS
from wrappers.siamfc_wrapper import SiamFCWrapper
from wrappers.siamrpnpp_wrapper import SiamRPNppWrapper

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'tracking_secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

tracking_state = {'running': False, 'stop': False}
frame_cache = {'images': [], 'pred_bboxes': [], 'gts': [], 'algo_name': '', 'total_frames': 0}

ALGORITHMS = {
    **{k: {'type': 'traditional', 'key': k, 'name': v['name']}
       for k, v in TRADITIONAL_METHODS.items()},
    'siamfc': {'type': 'deep', 'key': 'siamfc', 'name': 'SiamFC'},
    'siamrpnpp': {'type': 'deep', 'key': 'siamrpnpp', 'name': 'SiamRPN++'},
}

def allowed_file(filename, exts=None):
    if exts is None: exts = config.ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def compute_iou(box1, box2):
    x1,y1,w1,h1=box1[:4]; x2,y2,w2,h2=box2[:4]
    xx1,yy1=max(x1,x2),max(y1,y2); xx2,yy2=min(x1+w1,x2+w2),min(y1+h1,y2+h2)
    inter=max(0,xx2-xx1)*max(0,yy2-yy1)
    return inter/max(w1*h1+w2*h2-inter,1e-10)

def compute_cle(box1, box2):
    return np.sqrt((box1[0]+box1[2]/2-box2[0]-box2[2]/2)**2+(box1[1]+box1[3]/2-box2[1]-box2[3]/2)**2)

def create_tracker(algo_key):
    if algo_key in TRADITIONAL_METHODS: return TraditionalTrackerWrapper(algo_key)
    elif algo_key=='siamfc': return SiamFCWrapper()
    elif algo_key=='siamrpnpp': return SiamRPNppWrapper()
    raise ValueError(f'未知算法: {algo_key}')

def load_first_frame(video_path):
    cap=cv2.VideoCapture(video_path)
    ret,frame=cap.read()
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ret: return None,0,0,0,0
    _,buf=cv2.imencode('.jpg',frame)
    return f'data:image/jpeg;base64,{base64.b64encode(buf).decode("utf-8")}',total,fps,w,h

def find_gt_file(video_path):
    base=os.path.splitext(video_path)[0]
    for c in [base+'_gt.txt',base+'.txt',os.path.join(os.path.dirname(video_path),'groundtruth_rect.txt')]:
        if os.path.exists(c):
            for d in [',', '\t', None]:
                try:
                    gt=np.loadtxt(c,delimiter=d)
                    if len(gt.shape)==1: gt=gt.reshape(1,-1)
                    return gt
                except: pass
    return None

@app.route('/')
def index(): return send_file('static/index.html')

@app.route('/api/algorithms')
def get_algorithms(): return jsonify(list(ALGORITHMS.values()))

@app.route('/api/otb_list')
def get_otb_list():
    otb_dir=os.path.join(os.path.dirname(__file__),'otb_videos')
    videos=[]
    if os.path.isdir(otb_dir):
        for name in sorted(os.listdir(otb_dir)):
            vp=os.path.join(otb_dir,name,f'{name}.mp4')
            gp=os.path.join(otb_dir,name,'groundtruth_rect.txt')
            if os.path.exists(vp) and os.path.exists(gp):
                cap=cv2.VideoCapture(vp)
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps=cap.get(cv2.CAP_PROP_FPS)
                w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                videos.append({'name':name,'path':vp,'gt':gp,'frames':total,'fps':round(fps,1),'w':w,'h':h})
    return jsonify(videos)

@app.route('/api/load_otb', methods=['POST'])
def load_otb():
    data=request.get_json()
    if not os.path.exists(data.get('path','')): return jsonify({'error':'文件不存在'}),400
    img,total,fps,w,h=load_first_frame(data['path'])
    if img is None: return jsonify({'error':'无法读取'}),400
    return jsonify({'filepath':data['path'],'first_frame':img,'total_frames':total,'fps':round(fps,1),'width':w,'height':h,'has_gt':True})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({'error':'未找到视频'}),400
    file=request.files['video']
    if file.filename=='' or not allowed_file(file.filename): return jsonify({'error':'不支持格式'}),400
    fn=secure_filename(file.filename)
    fp=os.path.join(config.UPLOAD_DIR,fn)
    file.save(fp)
    img,total,fps,w,h=load_first_frame(fp)
    if img is None: return jsonify({'error':'无法读取'}),400
    return jsonify({'filename':fn,'filepath':fp,'total_frames':total,'fps':round(fps,1),'width':w,'height':h,'first_frame':img,'has_gt':False})

@app.route('/api/upload_gt', methods=['POST'])
def upload_gt():
    if 'gt' not in request.files: return jsonify({'error':'未找到GT文件'}),400
    gf=request.files['gt']; vp=request.form.get('video_path','')
    if not vp: return jsonify({'error':'缺少视频路径'}),400
    gp=os.path.splitext(vp)[0]+'_gt.txt'; gf.save(gp)
    for d in [',', '\t', None]:
        try:
            gt=np.loadtxt(gp,delimiter=d)
            if len(gt.shape)==1: gt=gt.reshape(1,-1)
            return jsonify({'message':f'加载{len(gt)}帧','frames':len(gt)})
        except: pass
    os.remove(gp); return jsonify({'error':'格式错误(需逗号或制表符分隔)'}),400

# ===== 帧跳转（处理后回放）=====
@socketio.on('seek_frame')
def handle_seek_frame(data):
    fid=data['frame_id']
    if fid<0 or fid>=len(frame_cache['images']):
        emit('error',{'message':'帧不存在'}); return
    emit('frame_seeked',{
        'frame_id':fid,
        'image':frame_cache['images'][fid],
        'pred_bbox':frame_cache['pred_bboxes'][fid] if fid<len(frame_cache['pred_bboxes']) else None,
        'gt_bbox':frame_cache['gts'][fid] if fid<len(frame_cache['gts']) else None,
        'algo_name':frame_cache.get('algo_name',''),
    })

# ===== 逐帧跟踪 =====
@socketio.on('start_tracking')
def handle_start_tracking(data):
    global tracking_state, frame_cache
    tracking_state['running']=True; tracking_state['stop']=False
    frame_cache={'images':[],'pred_bboxes':[],'gts':[],'algo_name':'','total_frames':0}

    vp=data['video_path']; ak=data['algorithm']; ib=data.get('init_bbox')
    hg=data.get('has_gt',False); auto_ib=data.get('auto_init_bbox',False)
    cap=cv2.VideoCapture(vp)
    tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt=find_gt_file(vp) if hg else None
    # OTB模式(auto_ib=True)：用GT第一帧作为初始框
    if auto_ib and hg and gt is not None and len(gt)>0:
        ib = [float(gt[0][0]), float(gt[0][1]), float(gt[0][2]), float(gt[0][3])]
        print(f'[tracking] OTB模式，使用GT第一帧作为初始框: {ib}')

    try: tracker=create_tracker(ak)
    except Exception as e: emit('error',{'message':f'创建跟踪器失败:{str(e)}'}); tracking_state['running']=False; return

    ai,ac,ft,fps_hist=[],[],[],[]
    fid=0
    while True:
        if tracking_state['stop']: break
        ret,frame=cap.read()
        if not ret: break
        ts=time.time()
        if fid==0: pb=tracker.init(frame,ib)
        else:
            try: pb=tracker.update(frame)
            except: break
        proc_time=time.time()-ts
        ft.append(proc_time)
        current_fps=round(1.0/max(proc_time,1e-6),1)
        fps_hist.append(current_fps)
        ci=cc=0.0
        if hg and gt is not None and fid<len(gt):
            ci=compute_iou(pb,gt[fid]); cc=compute_cle(pb,gt[fid])
            ai.append(ci); ac.append(cc)
        display=frame.copy()
        p=[int(v) for v in pb]
        cv2.rectangle(display,(p[0],p[1]),(p[0]+p[2],p[1]+p[3]),(0,255,255),3)
        if hg and gt is not None and fid<len(gt):
            g=[int(v) for v in gt[fid]]
            cv2.rectangle(display,(g[0],g[1]),(g[0]+g[2],g[1]+g[3]),(0,0,255),3)
        cv2.putText(display,f'帧{fid}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(display,tracker.get_name(),(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        _,buf=cv2.imencode('.jpg',display,[cv2.IMWRITE_JPEG_QUALITY,80])
        ib64=f'data:image/jpeg;base64,{base64.b64encode(buf).decode("utf-8")}'
        frame_cache['images'].append(ib64); frame_cache['pred_bboxes'].append(pb)
        if hg and gt is not None and fid<len(gt): frame_cache['gts'].append(gt[fid].tolist())
        else: frame_cache['gts'].append(None)
        frame_cache['algo_name']=tracker.get_name(); frame_cache['total_frames']=tf
        # 时间轴缩略图（每15帧发一张）
        tl_thumb=None
        if fid%15==0:
            _,tb=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,30])
            tl_thumb=f'data:image/jpeg;base64,{base64.b64encode(tb).decode("utf-8")}'
        emit('frame_result',{'frame_id':fid,'total_frames':tf,'image':ib64,'pred_bbox':pb,
             'iou':round(ci,4) if hg else None,'cle':round(cc,2) if hg else None,
             'fps':current_fps,'has_gt':hg,'thumb':tl_thumb},broadcast=True)
        socketio.sleep(0.001); fid+=1
    cap.release(); tracking_state['running']=False
    tt=sum(ft); afps=fid/max(tt,1e-6) if fid>1 else 0
    summary={'total_frames':fid,'avg_fps':round(afps,1),'algo_name':tracker.get_name(),'fps_history':fps_hist}
    if hg and len(ai)>0:
        si=np.sort(ai)
        summary.update({'avg_iou':round(np.mean(ai),4),'avg_cle':round(np.mean(ac),2),
            'success_rate':round(np.mean(np.array(ai)>0.5),4),
            'precision':round(np.mean(np.array(ac)<=20),4),'auc':round(np.trapz(si,np.linspace(0,1,len(si))),4)})
    emit('tracking_done',summary,broadcast=True)

@socketio.on('stop_tracking')
def handle_stop_tracking(): tracking_state['stop']=True

@socketio.on('connect')
def handle_connect(): emit('connected',{'message':'已连接'})

if __name__=='__main__':
    os.makedirs(config.UPLOAD_DIR,exist_ok=True)
    os.makedirs(config.RESULT_DIR,exist_ok=True)
    socketio.run(app,host='0.0.0.0',port=5000,debug=False,allow_unsafe_werkzeug=True)
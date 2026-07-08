const socket=io();
let isTracking=false,trackingDone=false,mode=null,hasGt=false;
let replayTimer=null,iouData=[],cleData=[],fpsData=[];
let upVideoInfo=null,upInitBbox=null,upBboxSel=false,upBboxStart=null,upLastBbox=null;
let otbVideoInfo=null,tlStrip=null;
const $=id=>document.getElementById(id);
const upC=$('upCanvas'),upCtx=upC.getContext('2d');

function showP(m){mode=m;hasGt=(m==='otb');
    $('pageSelect').style.display='none';$('modalUpload').style.display='none';$('modalOtb').style.display='none';
    $('playerUpload').style.display=m==='upload'?'block':'none';$('playerOtb').style.display=m==='otb'?'block':'none';}
window.openModal=function(t){if(t==='upload')resetUp();else resetOtb();$(t==='upload'?'modalUpload':'modalOtb').style.display='flex';};
window.closeModal=function(t){$(t==='upload'?'modalUpload':'modalOtb').style.display='none';};
window.backToSelect=function(){if(isTracking)socket.emit('stop_tracking');if(replayTimer){clearInterval(replayTimer);replayTimer=null;}
    isTracking=false;trackingDone=false;$('pageSelect').style.display='flex';$('playerUpload').style.display='none';$('playerOtb').style.display='none';resetUp();resetOtb();};
async function init(){await loadAlgorithms();await loadOtbList();bindEvents();}
async function loadAlgorithms(){
    try{const r=await fetch('/api/algorithms');const algos=await r.json();
        ['upAlgoSelect','otbAlgoSelect'].forEach(id=>{
            const sel=$(id);sel.innerHTML='<option value="">-- 选择算法 --</option>';
            algos.forEach(a=>{const o=document.createElement('option');o.value=a.key;o.textContent=a.name+(a.type==='deep'?'🧠':'📐');sel.appendChild(o);});sel.disabled=false;});}
    catch(e){alert('❌ 无法加载算法: '+e.message);}}
async function loadOtbList(){
    try{const r=await fetch('/api/otb_list');const videos=await r.json();
        const sel=$('otbVideoSelect');sel.innerHTML='<option value="">-- 选择OTB视频 --</option>';
        videos.forEach(v=>{const o=document.createElement('option');o.value=JSON.stringify(v);o.textContent=v.name+' ('+v.frames+'帧, '+v.w+'x'+v.h+')';sel.appendChild(o);});
        $('otbStatus').textContent=videos.length+'个视频';$('otbStatus').style.color='#2ecc71';}
    catch(e){alert('❌ 无法加载OTB列表: '+e.message);}}
function bindEvents(){
    $('upUploadBtn').addEventListener('click',uploadVideo);
    $('upBboxBtn').addEventListener('click',()=>{if(!upVideoInfo)return;upBboxSel=true;upBboxStart=null;$('upBboxBtn').textContent='拖动';$('upCancelBboxBtn').style.display='inline-block';});
    $('upCancelBboxBtn').addEventListener('click',cancelBbox);
    $('upStartBtn').addEventListener('click',()=>startTrack('upload'));
    $('upAlgoSelect').addEventListener('change',()=>{$('upStartBtn').disabled=!($('upAlgoSelect').value&&upInitBbox&&upVideoInfo);});
    upC.addEventListener('mousedown',e=>{if(upBboxSel)upBboxStart=getPos(e,upC);});
    upC.addEventListener('mousemove',e=>{if(!upBboxSel||!upBboxStart)return;const p=getPos(e,upC);if(upVideoInfo){const img=new Image();img.onload=()=>{upCtx.clearRect(0,0,upC.width,upC.height);upCtx.drawImage(img,0,0);const x=Math.min(upBboxStart.x,p.x),y=Math.min(upBboxStart.y,p.y),w=Math.abs(p.x-upBboxStart.x),h=Math.abs(p.y-upBboxStart.y);upCtx.strokeStyle='#ff0';upCtx.lineWidth=2;upCtx.strokeRect(x,y,w,h);upCtx.fillStyle='rgba(255,255,0,0.2)';upCtx.fillRect(x,y,w,h);};img.src=upVideoInfo.first_frame;}});
    upC.addEventListener('mouseup',e=>{if(!upBboxSel||!upBboxStart)return;const p=getPos(e,upC);const x=Math.min(upBboxStart.x,p.x),y=Math.min(upBboxStart.y,p.y),w=Math.abs(p.x-upBboxStart.x),h=Math.abs(p.y-upBboxStart.y);if(w>5&&h>5){upInitBbox=[Math.round(x),Math.round(y),Math.round(w),Math.round(h)];upLastBbox=[x,y,w,h];$('upBboxStatus').textContent='✅ ['+upInitBbox.join(',')+']';upBboxSel=false;$('upBboxBtn').textContent='框选';$('upCancelBboxBtn').style.display='none';$('upStartBtn').disabled=!($('upAlgoSelect').value&&upInitBbox);drawUpFrame();}});
    $('otbLoadBtn').addEventListener('click',loadOtb);
    $('otbStartBtn').addEventListener('click',()=>startTrack('otb'));
    $('otbAlgoSelect').addEventListener('change',()=>{$('otbStartBtn').disabled=!($('otbAlgoSelect').value&&otbVideoInfo);});
    document.addEventListener('keydown',e=>{if(e.key==='Escape')cancelBbox();});
    $('puTLR').addEventListener('input',()=>onTL('pu'));$('poTLR').addEventListener('input',()=>onTL('po'));}
function getPos(e,c){const r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height};}
function drawUpFrame(){if(upVideoInfo){const img=new Image();img.onload=()=>{upCtx.clearRect(0,0,upC.width,upC.height);upCtx.drawImage(img,0,0);if(upLastBbox){const [x,y,w,h]=upLastBbox;upCtx.strokeStyle='#ff0';upCtx.lineWidth=3;upCtx.strokeRect(x,y,w,h);upCtx.fillStyle='rgba(255,255,0,0.2)';upCtx.fillRect(x,y,w,h);}};img.src=upVideoInfo.first_frame;}}
function cancelBbox(){upBboxSel=false;upBboxStart=null;upInitBbox=null;upLastBbox=null;$('upBboxBtn').textContent='框选';$('upCancelBboxBtn').style.display='none';$('upBboxStatus').textContent='未选择';drawUpFrame();}
function resetUp(){upVideoInfo=null;upInitBbox=null;upBboxSel=false;upBboxStart=null;upLastBbox=null;$('upVideoStatus').textContent='未选择';$('upBboxStatus').textContent='未选择';$('upHint').style.display='flex';upCtx.clearRect(0,0,upC.width,upC.height);$('upBboxBtn').disabled=true;$('upStartBtn').disabled=true;}
function resetOtb(){otbVideoInfo=null;$('otbStatus').textContent='未选择';$('otbHint').style.display='flex';$('otbCanvas').getContext('2d').clearRect(0,0,$('otbCanvas').width,$('otbCanvas').height);$('otbStartBtn').disabled=true;}
async function uploadVideo(){
    const file=$('upVideoInput').files[0];if(!file)return alert('请选择视频文件');
    const fd=new FormData();fd.append('video',file);
    $('upUploadBtn').disabled=true;$('upUploadBtn').textContent='上传中…';
    try{const r=await fetch('/api/upload',{method:'POST',body:fd});const d=await r.json();
        if(d.error)return alert(d.error);upVideoInfo=d;$('upVideoStatus').textContent='✅ '+d.filename;$('upVideoStatus').style.color='#2ecc71';$('upHint').style.display='none';
        upC.width=d.width;upC.height=d.height;const img=new Image();img.onload=()=>{upCtx.clearRect(0,0,d.width,d.height);upCtx.drawImage(img,0,0);$('upBboxBtn').disabled=false;};img.src=d.first_frame;}
    catch(e){alert('上传失败: '+e.message);}finally{$('upUploadBtn').disabled=false;$('upUploadBtn').textContent='上传';}}
async function loadOtb(){
    const val=$('otbVideoSelect').value;if(!val)return alert('请选择OTB视频');
    const info=JSON.parse(val);$('otbLoadBtn').disabled=true;$('otbLoadBtn').textContent='加载中…';
    try{const r=await fetch('/api/load_otb',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:info.path})});const d=await r.json();
        if(d.error)return alert(d.error);otbVideoInfo=d;
        const c=$('otbCanvas');c.width=d.width;c.height=d.height;const ctx=c.getContext('2d');
        const img=new Image();img.onload=()=>{ctx.clearRect(0,0,d.width,d.height);ctx.drawImage(img,0,0);};img.src=d.first_frame;
        $('otbHint').style.display='none';$('otbStatus').textContent='✅ '+info.name;$('otbStatus').style.color='#2ecc71';
        $('otbStartBtn').disabled=!($('otbAlgoSelect').value);}
    catch(e){alert('加载失败: '+e.message);}finally{$('otbLoadBtn').disabled=false;$('otbLoadBtn').textContent='加载';}}
function startTrack(m){
    let vp,algo,bbox;
    if(m==='upload'){if(!upVideoInfo||!upInitBbox||!$('upAlgoSelect').value)return;
        vp=upVideoInfo.filepath;algo=$('upAlgoSelect').value;bbox=upInitBbox;}
    else{if(!otbVideoInfo||!$('otbAlgoSelect').value)return;
        vp=otbVideoInfo.filepath;algo=$('otbAlgoSelect').value;bbox=null;}
    isTracking=true;trackingDone=false;mode=m;hasGt=(m==='otb');
    iouData=[];cleData=[];fpsData=[];tlStrip=null;
    const pf=(m==='upload'?'pu':'po');const at=$('upAlgoSelect').options[$('upAlgoSelect').selectedIndex]?.text||$('otbAlgoSelect').options[$('otbAlgoSelect').selectedIndex]?.text||'';
    $(pf+'Title').textContent='⏳ '+at+' - 跟踪中...';$(pf+'Algo').textContent='算法: '+at;
    const vi=m==='upload'?upVideoInfo:otbVideoInfo;const pc=$(pf+'Canvas');pc.width=vi.width||640;pc.height=vi.height||480;
    $(pf+'Frame').textContent='帧: 0/'+(vi.total_frames||0);$(pf+'Summary').style.display='none';$(pf+'TLS_').textContent='⏳ 处理中...';$(pf+'TLR').disabled=true;
    showP(m);
    // 创建tlStrip（flex布局，始终可见）
    const track=$(pf+'TL').parentElement;
    if(track){
        const old=track.querySelector('.tl-strip');
        if(old)old.remove();
        const div=document.createElement('div');div.className='tl-strip';
        div.style.cssText='position:absolute;top:0;left:0;width:100%;height:50px;display:flex;align-items:center;overflow:hidden;z-index:10;pointer-events:none;';
        track.style.position='relative';track.appendChild(div);tlStrip=div;
    }
    try{window.poIou&&window.poIou.destroy()}catch(e){}try{window.poCle&&window.poCle.destroy()}catch(e){}try{window.poFps&&window.poFps.destroy()}catch(e){}try{window.puFps&&window.puFps.destroy()}catch(e){}
    if(m==='otb'){
        window.poIou=new Chart($('poIouChart'),{type:'line',data:{labels:[],datasets:[{label:'IoU',data:[],borderColor:'#2ecc71',backgroundColor:'rgba(46,204,113,0.1)',borderWidth:2,pointRadius:1,fill:true}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:true,ticks:{font:{size:9}}},y:{min:0,max:1,ticks:{font:{size:10}}}},plugins:{legend:{display:false}}}});
        window.poCle=new Chart($('poCleChart'),{type:'line',data:{labels:[],datasets:[{label:'CLE',data:[],borderColor:'#e74c3c',backgroundColor:'rgba(231,76,60,0.1)',borderWidth:2,pointRadius:1,fill:true}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:true,ticks:{font:{size:9}}},y:{min:0,ticks:{font:{size:10}}}},plugins:{legend:{display:false}}}});
        window.poFps=new Chart($('poFpsChart'),{type:'line',data:{labels:[],datasets:[{label:'FPS',data:[],borderColor:'#f39c12',backgroundColor:'rgba(243,156,18,0.1)',borderWidth:2,pointRadius:1,fill:true}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:true,ticks:{font:{size:9}}},y:{min:0,ticks:{font:{size:10}}}},plugins:{legend:{display:false}}}});}
    else{window.puFps=new Chart($('puFpsChart'),{type:'line',data:{labels:[],datasets:[{label:'FPS',data:[],borderColor:'#f39c12',backgroundColor:'rgba(243,156,18,0.1)',borderWidth:2,pointRadius:1,fill:true}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:true,ticks:{font:{size:9}}},y:{min:0,ticks:{font:{size:10}}}},plugins:{legend:{display:false}}}});}
    socket.emit('start_tracking',{video_path:vp,algorithm:algo,init_bbox:bbox,has_gt:hasGt,auto_init_bbox:m==='otb'});}
function onTL(p){if(!trackingDone){$(p+'TLS_').textContent='⏳ 处理中...';return;}
    if(replayTimer){clearInterval(replayTimer);replayTimer=null;}let rf=parseInt($(p+'TLR').value);$(p+'TLS_').textContent='▶ 回放';
    socket.emit('seek_frame',{frame_id:rf});
    replayTimer=setInterval(()=>{rf++;if(rf>=parseInt($(p+'TLR').max)){clearInterval(replayTimer);replayTimer=null;$(p+'TLS_').textContent='✅ 回放结束';return;}$(p+'TLR').value=rf;$(p+'TLL').textContent='帧 '+rf;socket.emit('seek_frame',{frame_id:rf});},50);}
function updateLiveChart(chart,fid,val){if(chart.data.labels.length>=30){chart.data.labels.shift();chart.data.datasets[0].data.shift();}
    chart.data.labels.push(fid);chart.data.datasets[0].data.push(val);chart.update('none');}
function makeFinalChart(canvasId,label,data,color,minY){
    const c=document.createElement('canvas');c.id=canvasId;c.style.cssText='width:100%;height:150px;display:block;';
    const div=document.createElement('div');div.style.cssText='background:#fff;border-radius:10px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:10px;max-height:200px;overflow:hidden;';
    const h3=document.createElement('h3');h3.style.fontSize='14px';h3.textContent='📈 '+label+'完整曲线';div.appendChild(h3);div.appendChild(c);
    new Chart(c,{type:'line',data:{labels:data.map((_,i)=>i),datasets:[{label:label,data:data,borderColor:color,backgroundColor:color+'22',borderWidth:2,pointRadius:0,fill:true}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{display:true,ticks:{font:{size:10}}},y:{min:minY||0,ticks:{font:{size:10}}}},plugins:{legend:{display:false}}}});return div;}
socket.on('frame_result',(data)=>{
    if(!isTracking)return;const p=mode==='upload'?'pu':'po';const pc=$(p+'Canvas'),pctx=pc.getContext('2d');
    $(p+'Frame').textContent='帧: '+data.frame_id+'/'+data.total_frames;
    $(p+'Fps').textContent='FPS: '+data.fps;$(p+'MFr').textContent=data.frame_id+'/'+data.total_frames;$(p+'MFps').textContent=data.fps+' fps';
    const img=new Image();img.onload=()=>{pctx.drawImage(img,0,0,pc.width,pc.height);};img.src=data.image;
    fpsData.push(data.fps);
    if(mode==='otb'&&data.has_gt&&data.iou!==null){iouData.push(data.iou);cleData.push(data.cle);
        $(p+'MIoU').textContent=data.iou.toFixed(4);$(p+'MCLE').textContent=data.cle.toFixed(1)+'px';$(p+'MIoUB').style.width=(data.iou*100)+'%';
        updateLiveChart(window.poIou,data.frame_id,data.iou);updateLiveChart(window.poCle,data.frame_id,data.cle);}
    if(mode==='otb')updateLiveChart(window.poFps,data.frame_id,data.fps);else updateLiveChart(window.puFps,data.frame_id,data.fps);
    // 缩略图添加到tlStrip（flex自动排列）
    if(data.thumb&&data.total_frames>0&&tlStrip){
        const step=800/data.total_frames;
        const ti=document.createElement('img');
        ti.src=data.thumb;
        ti.style.cssText='height:46px;width:'+Math.max(step,2)+'px;object-fit:cover;flex-shrink:0;';
        ti.title='帧 '+data.frame_id;
        tlStrip.appendChild(ti);
    }
    $(p+'TLR').max=data.total_frames-1;$(p+'TLR').value=data.frame_id;
    $(p+'TLL').textContent='帧 '+data.frame_id;$(p+'TLT').textContent='/ '+data.total_frames;});
socket.on('tracking_done',(summary)=>{console.log('tracking_done:',JSON.stringify(summary).slice(0,200));
    isTracking=false;trackingDone=true;const p=mode==='upload'?'pu':'po';
    $(p+'TLS_').textContent='✅ 完成，可拖动';$(p+'TLR').disabled=false;
    $(p+'Title').textContent='✅ 完成 - '+summary.algo_name;$(p+'Summary').style.display='block';
    let f=[['算法',summary.algo_name],['处理帧数',summary.total_frames+'帧'],['平均FPS',summary.avg_fps+'fps']];
    if(summary.avg_iou!==undefined){f.push(['平均IoU',summary.avg_iou.toFixed(4)],['平均CLE',summary.avg_cle.toFixed(1)+'px'],['成功率',(summary.success_rate*100).toFixed(1)+'%'],['Precision',(summary.precision*100).toFixed(1)+'%'],['AUC',summary.auc.toFixed(4)]);}
    $(p+'Grid').innerHTML=f.map(x=>'<div class="summary-item"><div class="label">'+x[0]+'</div><div class="value">'+x[1]+'</div></div>').join('');
    const fc=$(p+'FinalCharts');fc.innerHTML='';
    if(fpsData.length>0){fc.appendChild(makeFinalChart(p+'FinFps','FPS',fpsData,'#f39c12',0));
        if(mode==='otb'&&iouData.length>0){fc.appendChild(makeFinalChart(p+'FinIou','IoU',iouData,'#2ecc71',0));fc.appendChild(makeFinalChart(p+'FinCle','CLE',cleData,'#e74c3c',0));}}});
socket.on('frame_seeked',(data)=>{const p=mode==='upload'?'pu':'po';const pc=$(p+'Canvas');
    const img=new Image();img.onload=()=>{pc.getContext('2d').drawImage(img,0,0,pc.width,pc.height);};img.src=data.image;
    $(p+'Algo').textContent='算法: '+(data.algo_name||'');});
socket.on('error',(data)=>{alert('错误: '+data.message);isTracking=false;});init();
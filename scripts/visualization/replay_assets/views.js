// Both cameras use the same interpolated trajectory and signal clock.
const env = DATA.environment;
const fp = document.getElementById('fp'), fc = fp.getContext('2d');
const followBox = document.getElementById('follow');
const labelsBox = document.getElementById('labels');
const vehicleSelect = document.getElementById('vehicle-select');
const signalColors = {red:'#f04452', yellow:'#f5bd35', green:'#26c980', unknown:'#8492a6'};
const turnGlyph = {left:'←', right:'→', straight:'↑', uturn:'↶', u_turn:'↶'};
const planByConn = new Map();
for (const plan of env.signal_plans) {
  plan.cycle = plan.phases.reduce((s,p)=>s+p.green_s+p.yellow_s+p.all_red_s,0);
  for (const phase of plan.phases) for (const cid of phase.connector_ids) planByConn.set(cid,plan);
}
function signalAt(cid,t) {
  const plan = planByConn.get(cid);
  if (!plan || !plan.cycle) return 'unknown';
  let clock = ((t%plan.cycle)+plan.cycle)%plan.cycle;
  for (const p of plan.phases) {
    const duration = p.green_s+p.yellow_s+p.all_red_s;
    if (clock < duration) return p.connector_ids.includes(cid)
      ? (clock<p.green_s?'green':clock<p.green_s+p.yellow_s?'yellow':'red') : 'red';
    clock -= duration;
  }
  return 'red';
}
function polygonOf(r) { return [...r.left_boundary_xy,...r.right_boundary_xy.slice().reverse()]; }
function bounded(points) {
  return {points, bounds:[Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1])),
    Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))]};
}
const roadSurfaces = [...env.lanes,...env.connectors].map(r=>bounded(polygonOf(r)));
const boundaries = env.lanes.flatMap(r=>[bounded(r.left_boundary_xy),bounded(r.right_boundary_xy)]);
const stripes = env.crosswalks.flatMap(c=>(c.stripe_polygons_xy?.length?c.stripe_polygons_xy:[c.polygon_xy]).filter(p=>p?.length).map(bounded));
const stops = env.stop_lines.map(s=>({...bounded(s.line_xy),id:s.id}));
// A separate head for each movement prevents a shared lane from hiding conflicting colors.
const groups = new Map();
for (const [cid,c] of Object.entries(conns)) {
  if (!c.sig) continue;
  const key = c.from+'|'+c.turn;
  if (!groups.has(key)) groups.set(key,{cid,cids:[],lane:c.from,turn:c.turn});
  groups.get(key).cids.push(cid);
}
const headsByLane = new Map();
for (const h of groups.values()) {
  if (!headsByLane.has(h.lane)) headsByLane.set(h.lane,[]);
  headsByLane.get(h.lane).push(h);
}
const heads = [];
for (const [lid,group] of headsByLane) {
  const pts = lanes[lid]; if (!pts || pts.length<2) continue;
  const a=pts.at(-2), b=pts.at(-1), angle=Math.atan2(b[1]-a[1],b[0]-a[0]);
  group.forEach((h,i)=>{
    const shift=(i-(group.length-1)/2)*1.15;
    heads.push({...h,x:b[0]-Math.sin(angle)*shift,y:b[1]+Math.cos(angle)*shift,angle});
  });
}
function headState(h) {
  const states=h.cids.map(cid=>signalAt(cid,tNow));
  return states.every(s=>s===states[0])?states[0]:'unknown';
}
const turnNames = {left:'左转',right:'右转',straight:'直行',uturn:'掉头',u_turn:'掉头'};
const stateNames = {red:'红灯',yellow:'黄灯',green:'绿灯',unknown:'未知'};
function approachSignals(ego) {
  if(!ego || ego[6]>=0)return [];
  // Lane identity, rather than nearest lamp, keeps opposing/cross traffic out of the HUD.
  return heads.filter(head=>head.lane===laneIds[ego[5]]).map(head=>({head,
    distance:Math.hypot(head.x-ego[1],head.y-ego[2]),state:headState(head)}))
    .filter(item=>item.distance<=120);
}
function drawSignalHousing(context,x,y,r,state) {
  context.fillStyle='#101c2c';context.fillRect(x-r*1.65,y-r*4.4,r*3.3,r*8.8);
  context.strokeStyle='#e1e8ef';context.lineWidth=Math.max(1,devicePixelRatio*.75);
  context.strokeRect(x-r*1.65,y-r*4.4,r*3.3,r*8.8);
  ['red','yellow','green'].forEach((color,i)=>{
    context.beginPath();context.arc(x,y+(i-1)*r*2.7,r,0,2*Math.PI);
    context.fillStyle=state===color?signalColors[color]:'#354052';context.fill();
    if(state===color){context.strokeStyle='#fff';context.lineWidth=Math.max(1,r*.12);context.stroke();}
  });
}
function drawApproachSignals(ego,w,h) {
  const items=approachSignals(ego);
  // Also present in canvas video exports; this is explicitly a replay annotation.
  const d=devicePixelRatio, left=14*d, top=78*d;
  const cardWidth=Math.min(w-28*d,Math.max(220*d,items.length*84*d+24*d));
  fc.fillStyle='#101c2ce8';fc.fillRect(left,top,cardWidth,(items.length?130:48)*d);
  fc.fillStyle='#e8f1fa';fc.font=`${12*d}px sans-serif`;
  fc.fillText('当前车道信号 · 配时还原',left+12*d,top+19*d);
  if(!items.length){fc.fillStyle='#aabacd';fc.font=`${11*d}px sans-serif`;
    fc.fillText(ego[6]>=0?'已进入路口':'前方 120 m 内无当前车道信号',left+12*d,top+37*d);return;}
  const cell=(cardWidth-24*d)/items.length;
  items.forEach((item,i)=>{
    const x=left+12*d+cell*(i+.5),y=top+66*d;
    drawSignalHousing(fc,x,y,5.5*d,item.state);
    fc.textAlign='center';fc.font=`bold ${12*d}px sans-serif`;fc.fillStyle=signalColors[item.state];
    fc.fillText((turnNames[item.head.turn]||item.head.turn)+' · '+stateNames[item.state],x,top+106*d);
  });
  fc.textAlign='start';fc.fillStyle='#aabacd';fc.font=`${10*d}px sans-serif`;
  fc.fillText('距信号入口约 '+Math.round(Math.min(...items.map(i=>i.distance)))+' m',left+12*d,top+122*d);
}
function path(context,points,project,closed=false) {
  if (!points.length) return;
  context.beginPath(); points.forEach((p,i)=>{const q=project(...p);i?context.lineTo(...q):context.moveTo(...q);});
  if(closed) context.closePath();
}
function visible(r,cx,cy,rx,ry=rx) {
  const b=r.bounds;return b[2]>=cx-rx && b[0]<=cx+rx && b[3]>=cy-ry && b[1]<=cy+ry;
}
let labelBoxes=[];
function mapLabel(text,x,y,color='#34465a') {
  const q=w2s(x,y);ctx.font=`${11*devicePixelRatio}px sans-serif`;
  const w=ctx.measureText(text).width;
  const box=[q[0]-3,q[1]-13*devicePixelRatio,q[0]+w+3,q[1]+3*devicePixelRatio];
  if(labelBoxes.some(b=>box[0]<b[2]+3&&box[2]>b[0]-3&&box[1]<b[3]+3&&box[3]>b[1]-3))return;
  labelBoxes.push(box);ctx.fillStyle='#ffffffdd';ctx.fillRect(box[0],box[1],w+6,16*devicePixelRatio);
  ctx.fillStyle=color;ctx.fillText(text,q[0],q[1]);
}
function pedestrianAt(rows,t) {
  if (!rows.length || t<rows[0][0] || t>rows.at(-1)[0]) return null;
  if(t===rows.at(-1)[0])return rows.at(-1)[3]===false?null:rows.at(-1).slice(1,3);
  let lo=0,hi=rows.length-1;
  while(hi-lo>1){const m=(lo+hi)>>1;if(rows[m][0]<=t)lo=m;else hi=m;}
  const a=rows[lo],b=rows[hi];if(a[3]===false)return null;
  if(b[3]===false)return a.slice(1,3);
  const k=Math.max(0,Math.min(1,(t-a[0])/(b[0]-a[0]||1)));
  return [a[1]+(b[1]-a[1])*k,a[2]+(b[2]-a[2])*k];
}
function drawEnvironment() {
  labelBoxes=[];
  ctx.fillStyle='#e1e8df';ctx.fillRect(0,0,cv.width,cv.height);
  const rx=cv.width/2/view.s,ry=cv.height/2/view.s;
  const local=r=>visible(r,view.cx,view.cy,rx,ry);
  ctx.fillStyle='#626e78';
  for(const r of roadSurfaces)if(local(r)){path(ctx,r.points,w2s,true);ctx.fill();}
  ctx.strokeStyle='#e3e6e2';ctx.lineWidth=Math.max(.65,view.s*.1);
  for(const r of boundaries)if(local(r)){path(ctx,r.points,w2s);ctx.stroke();}
  ctx.fillStyle='#f2f0e6';
  for(const r of stripes)if(local(r)){path(ctx,r.points,w2s,true);ctx.fill();}
  ctx.strokeStyle='#fff';ctx.lineWidth=Math.max(1.3,view.s*.4);
  for(const r of stops)if(local(r)){path(ctx,r.points,w2s);ctx.stroke();}
  if(document.getElementById('route').checked && selected){
    ctx.strokeStyle='#45acffb0';ctx.lineWidth=2*devicePixelRatio;
    path(ctx,vehicles[selected].filter(f=>f[7]!==false).map(f=>[f[1],f[2]]),w2s);ctx.stroke();
  }
  for(const h of heads){
    const q=w2s(h.x,h.y);if(q[0]<0||q[1]<0||q[0]>cv.width||q[1]>cv.height)continue;
    const r=Math.max(2*devicePixelRatio,Math.min(9*devicePixelRatio,view.s*.48));
    ctx.beginPath();ctx.arc(...q,r,0,2*Math.PI);ctx.fillStyle=signalColors[headState(h)];ctx.fill();
    ctx.strokeStyle='#273345';ctx.lineWidth=devicePixelRatio;ctx.stroke();
    if(view.s>5*devicePixelRatio){ctx.fillStyle='#142331';ctx.font=`bold ${r*1.6}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(turnGlyph[h.turn]||'↑',...q);ctx.textAlign='start';ctx.textBaseline='alphabetic';}
  }
  if(labelsBox.checked && view.s>1.5*devicePixelRatio){
    for(const c of env.crosswalks)if(c.center_xy && Math.hypot(c.center_xy[0]-view.cx,c.center_xy[1]-view.cy)<Math.max(rx,ry))mapLabel('人行横道',...c.center_xy);
    for(const r of stops)if(local(r))mapLabel('停止线',...r.points[0]);
  }
  for(const [pid,rows] of Object.entries(DATA.pedestrians)){
    const p=pedestrianAt(rows,tNow);if(!p)continue;
    const q=w2s(...p);ctx.beginPath();ctx.arc(...q,Math.max(3,view.s*.3),0,7);ctx.fillStyle='#ba57cf';ctx.fill();
    if(labelsBox.checked)mapLabel(pid,p[0]+1,p[1]+1,'#87389b');
  }
}
function focusVehicle(){
  const f=frameAt(selected,tNow);if(!f)return;
  view.cx=f[1];view.cy=f[2];view.s=Math.min(cv.width,cv.height)/110;
}
function updateFollow(){
  vehicleSelect.value=selected;
  if(followBox.checked){const f=frameAt(selected,tNow);if(f){view.cx=f[1];view.cy=f[2];}}
}
for(const vid of vids){const o=document.createElement('option');o.value=vid;o.textContent=vid;vehicleSelect.appendChild(o);}
vehicleSelect.value=selected;
vehicleSelect.onchange=()=>{selected=vehicleSelect.value;focusVehicle();};
document.getElementById('focus').onclick=()=>{followBox.checked=true;focusVehicle();};
document.getElementById('fit').onclick=()=>{followBox.checked=false;view={cx:(bx0+bx1)/2,cy:(by0+by1)/2,s:Math.min(cv.width/(bx1-bx0+80),cv.height/(by1-by0+80))};};
let panDistance=0,panStart=null;
cv.addEventListener('mousedown',e=>{panStart=[e.clientX,e.clientY];panDistance=0;});
addEventListener('mousemove',e=>{if(!drag||!panStart)return;panDistance=Math.max(panDistance,Math.hypot(e.clientX-panStart[0],e.clientY-panStart[1]));if(panDistance>4)followBox.checked=false;});
function inspectSignal(mx,my){
  let hit=null,dist=14*devicePixelRatio;
  for(const h of heads){const q=w2s(h.x,h.y),d=Math.hypot(q[0]-mx,q[1]-my);if(d<dist){hit=h;dist=d;}}
  if(hit){const box=document.getElementById('cmp');box.style.display='block';box.textContent='信号灯 · 地图配时还原\n车道：'+hit.lane+'\n转向：'+hit.turn+'\n'+hit.cids.map(cid=>cid+': '+signalAt(cid,tNow)).join('\n');box.style.whiteSpace='pre-wrap';}
}
// A ground-plane perspective camera with near-plane clipping. World +left maps to screen -x.
function drawFirstPerson(){
  const width=fp.clientWidth*devicePixelRatio,height=fp.clientHeight*devicePixelRatio;
  if(fp.width!==width||fp.height!==height){fp.width=width;fp.height=height;}
  const w=fp.width,h=fp.height,horizon=h*.42,focal=Math.min(w,h*1.7)*.84;
  const sky=fc.createLinearGradient(0,0,0,horizon);sky.addColorStop(0,'#7399bb');sky.addColorStop(1,'#dce5e7');fc.fillStyle=sky;fc.fillRect(0,0,w,horizon);
  fc.fillStyle='#bac4b2';fc.fillRect(0,horizon,w,h-horizon);
  const ego=frameAt(selected,tNow);
  document.getElementById('fp-status').textContent=selected+(ego?' · '+ego[4].toFixed(1)+' km/h':' · 当前时刻无轨迹');
  if(!ego){fc.fillStyle='#34465a';fc.font=`${16*devicePixelRatio}px sans-serif`;fc.textAlign='center';fc.fillText('当前时刻车辆不在场，请切换车辆或时间',w/2,h*.6);fc.textAlign='start';return;}
  const ca=Math.cos(ego[3]),sa=Math.sin(ego[3]);
  function camera(p){const dx=p[0]-ego[1],dy=p[1]-ego[2];return [dx*ca+dy*sa,dx*sa-dy*ca,p[2]||0];}
  function project(p){return [w/2+focal*p[1]/p[0],horizon+focal*(1.55-p[2])/p[0]];}
  function clipped(points,closed){
    const src=points.map(camera),out=[];
    for(let i=0;i<src.length;i++){
      const a=src[i];if(a[0]>=.5)out.push(a);
      if(!closed&&i===src.length-1)continue;
      const b=src[(i+1)%src.length];
      if((a[0]<.5)!==(b[0]<.5)){const k=(.5-a[0])/(b[0]-a[0]);out.push([.5,a[1]+k*(b[1]-a[1]),a[2]+k*(b[2]-a[2])]);}
    }
    return out.map(project);
  }
  function ground(points,color,closed=true,lineWidth=1){
    const ps=clipped(points,closed);if(ps.length<(closed?3:2))return;
    path(fc,ps,(x,y)=>[x,y],closed);if(closed){fc.fillStyle=color;fc.fill();}else{fc.strokeStyle=color;fc.lineWidth=lineWidth;fc.stroke();}
  }
  const near=r=>visible(r,ego[1],ego[2],160);
  for(const r of roadSurfaces)if(near(r))ground(r.points,'#525e68');
  for(const r of boundaries)if(near(r))ground(r.points,'#d6dedb',false,devicePixelRatio);
  for(const r of stripes)if(near(r))ground(r.points,'#efeee3');
  for(const r of stops)if(near(r))ground(r.points,'#fff',false,2*devicePixelRatio);
  const objects=[];
  for(const vid of vids){
    if(vid===selected)continue;const f=frameAt(vid,tNow);if(!f)continue;
    const p=camera([f[1],f[2]]);if(p[0]<-5||p[0]>160)continue;
    objects.push({depth:p[0],draw:()=>{
      const [length,width]=DATA.vehicle_dimensions[vid]||[4.6,1.9],c=Math.cos(f[3]),s=Math.sin(f[3]);
      const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>[f[1]+x*length/2*c-y*width/2*s,f[2]+x*length/2*s+y*width/2*c]);
      const faces=[];
      for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4];const points=[[...a,0],[...b,0],[...b,1.5],[...a,1.5]];faces.push({points,depth:points.reduce((d,p)=>d+camera(p)[0],0)/4,color:i%2?'#4b6179':'#6c8195'});}
      faces.push({points:corners.map(p=>[...p,1.5]),depth:p[0],color:'#90a4b7'});
      faces.sort((a,b)=>b.depth-a.depth).forEach(face=>ground(face.points,face.color));
    }});
  }
  for(const rows of Object.values(DATA.pedestrians)){
    const p=pedestrianAt(rows,tNow);if(!p)continue;const q=camera(p);if(q[0]<.5||q[0]>120)continue;
    objects.push({depth:q[0],draw:()=>{const feet=project(q),head=project([q[0],q[1],1.65]),r=Math.max(2,focal*.2/q[0]);fc.strokeStyle='#b557ce';fc.lineWidth=r*2;fc.beginPath();fc.moveTo(...feet);fc.lineTo(...head);fc.stroke();fc.beginPath();fc.arc(...head,r,0,7);fc.fillStyle='#f1c7a4';fc.fill();}});
  }
  for(const head of heads){
    const q=camera([head.x,head.y,4.8]);
    // Show only heads facing the camera's direction of travel.
    if(q[0]<1||q[0]>110||Math.cos(head.angle-ego[3])<.25)continue;
    objects.push({depth:q[0],draw:()=>{
      const top=project(q),bottom=project([q[0],q[1],0]);
      fc.strokeStyle='#344554';fc.lineWidth=Math.max(1,focal*.07/q[0]);fc.beginPath();fc.moveTo(...bottom);fc.lineTo(...top);fc.stroke();
      const r=Math.max(4*devicePixelRatio,Math.min(16*devicePixelRatio,focal*.3/q[0]));
      drawSignalHousing(fc,...top,r,headState(head));
      fc.fillStyle='#fff';fc.textAlign='center';fc.font=`bold ${Math.max(11*devicePixelRatio,r*1.8)}px sans-serif`;
      fc.fillText(turnGlyph[head.turn]||'↑',top[0],top[1]-r*5);fc.textAlign='start';
    }});
  }
  objects.sort((a,b)=>b.depth-a.depth).forEach(o=>o.draw());
  // A small dashboard gives a stable visual reference without covering the road.
  fc.fillStyle='#203044';fc.beginPath();fc.moveTo(0,h);fc.lineTo(0,h*.94);fc.quadraticCurveTo(w/2,h*.84,w,h*.94);fc.lineTo(w,h);fc.fill();
  fc.fillStyle='#e3edf6';fc.font=`${13*devicePixelRatio}px sans-serif`;fc.textAlign='right';fc.fillText(tNow.toFixed(1)+' s',w-18*devicePixelRatio,h-20*devicePixelRatio);fc.textAlign='start';
  drawApproachSignals(ego,w,h);
}
const resizeObserver=new ResizeObserver(()=>{resize();});resizeObserver.observe(document.getElementById('map-panel'));
focusVehicle();
// Optional browser recording keeps this HTML portable: no server or encoder install.
let recorder=null,recordChunks=[],recordStream=null;
const recordButton=document.getElementById('record'),recordStatus=document.getElementById('record-status');
if(!fp.captureStream || typeof MediaRecorder==='undefined'){recordButton.disabled=true;recordStatus.textContent='此浏览器不支持录制，请使用 Chrome / Edge';}
recordButton.onclick=()=>{
  if(recorder?.state==='recording'){recorder.stop();return;}
  const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4'].find(t=>MediaRecorder.isTypeSupported(t));
  if(!mime){recordStatus.textContent='此浏览器没有可用的视频编码器';return;}
  try{
    recordStream=fp.captureStream(30);recordChunks=[];recorder=new MediaRecorder(recordStream,{mimeType:mime});
    recorder.ondataavailable=e=>{if(e.data.size)recordChunks.push(e.data);};
    recorder.onstop=()=>{const blob=new Blob(recordChunks,{type:mime}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=DATA.meta.variant_id+'.'+selected+'.fpv.'+(mime.includes('mp4')?'mp4':'webm');a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);recordStream.getTracks().forEach(t=>t.stop());recordButton.textContent='录制第一视角';recordStatus.textContent='视频已导出';};
    recorder.start();recordButton.textContent='停止并下载';recordStatus.textContent='正在录制当前视角（按播放速度）';
    if(tNow>=tMax)tNow=tMin;playing=true;document.getElementById('play').textContent='⏸';
  }catch(error){recordStream?.getTracks().forEach(t=>t.stop());recordStatus.textContent='录制失败：'+error.message;}
};

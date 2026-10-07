"use strict";
const $=id=>document.getElementById(id),ns="http://www.w3.org/2000/svg";
let data=null,current=0,wave=0,playing=false,timer,inspected=0;
const maps={dense:[],fsd:[]},count=value=>value.toLocaleString("en-US");
const reportHeight=()=>{if(parent!==window)parent.postMessage({type:"fsd-animation-size",height:Math.ceil(document.body.getBoundingClientRect().height)},"*")};
new ResizeObserver(reportHeight).observe(document.body);addEventListener("load",reportHeight);
function element(tag,attrs={},text){const node=document.createElementNS(ns,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;return node;}
function asset(path){const url=new URL(path,new URL("../",location.href));if(url.origin!==location.origin||!/^https?:$/.test(url.protocol))throw new Error("Invalid recorded image source");return url.href;}
function picture(source){if(/^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(source))return source;return asset(source);}
function validate(input){
  const side=Number(input.grid),total=side*side;
  if(!Number.isInteger(side)||side<1||side>64||!Array.isArray(input.points)||input.points.length!==total)throw new Error("Point-grid coordinates are unavailable");
  for(let i=0;i<total;i++){const point=input.points[i];if(!Array.isArray(point)||point.length!==2||point.some(n=>!Number.isFinite(n)||n<0||n>1)||Math.abs(point[0]-((i%side)+.5)/side)>1e-7||Math.abs(point[1]-(Math.floor(i/side)+.5)/side)>1e-7)throw new Error("Invalid spatial point grid");}
  const validIds=ids=>Array.isArray(ids)&&new Set(ids).size===ids.length&&ids.every(i=>Number.isInteger(i)&&i>=0&&i<total);
  if(!validIds(input.selected)||!validIds(input.guards)||!validIds(input.guard_only)||!Array.isArray(input.waves)||input.waves.length!==4||!input.waves.every(validIds))throw new Error("Invalid selection coordinates");
  const nms=new Set(input.waves.flat()),guards=new Set(input.guards),selected=new Set(input.selected),guardOnly=new Set(input.guard_only),union=new Set([...nms,...guards]);
  if(nms.size!==input.waves.flat().length||selected.size!==union.size||[...selected].some(i=>!union.has(i))||guardOnly.size!==[...guards].filter(i=>!nms.has(i)).length||[...guardOnly].some(i=>!guards.has(i)||nms.has(i)))throw new Error("Selection counts do not match their point positions");
  const [height,width]=input.image_hw||[];if(![height,width].every(n=>Number.isInteger(n)&&n>0&&n<=10000))throw new Error("Image coordinates are unavailable");
  return {...input,grid:side,total,width,height,selectedSet:selected,guardSet:guards,guardOnlySet:guardOnly,nmsSet:nms};
}
function pointDescription(index){const row=Math.floor(index/data.grid)+1,col=index%data.grid+1,[x,y]=data.points[index],selectionWave=data.waves.findIndex(ids=>ids.includes(index));const reason=data.guardOnlySet.has(index)?"local guard":selectionWave>=0?`selection wave ${selectionWave+1}${data.guardSet.has(index)?" + local guard":""}`:"skip full decoding";return {position:`Point ${index+1} · row ${row}, column ${col} · x ${(x*data.width).toFixed(1)}, y ${(y*data.height).toFixed(1)} px`,reason};}
function buildMaps(){
  const radius=Math.min(data.width,data.height)/(data.grid*4.25),stroke=Math.max(.8,radius*.38);
  for(const method of ["dense","fsd"]){
    const svg=$(method+"-map");svg.replaceChildren();svg.setAttribute("viewBox",`0 0 ${data.width} ${data.height}`);
    svg.append(element("image",{width:data.width,height:data.height,href:picture(data.image),preserveAspectRatio:"none",class:"map-image"}));svg.append(element("rect",{width:data.width,height:data.height,fill:"white",opacity:.35}));
    const group=element("g",{class:"grid-points"});maps[method]=[];
    data.points.forEach((point,index)=>{const dot=element("circle",{cx:point[0]*data.width,cy:point[1]*data.height,r:radius,"stroke-width":stroke,class:"prompt-point","data-point":index});dot.append(element("title",{},pointDescription(index).position+" · "+(method==="dense"?"full native decoding":pointDescription(index).reason)));dot.addEventListener("click",()=>{inspected=index;inspectPoint();});group.append(dot);maps[method].push(dot);});
    svg.append(group);svg.append(element("circle",{class:"point-highlight",r:radius*1.95,"stroke-width":stroke*1.7}));
    svg.onclick=event=>{const point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;const local=point.matrixTransform(svg.getScreenCTM().inverse());const col=Math.max(0,Math.min(data.grid-1,Math.floor(local.x/data.width*data.grid))),row=Math.max(0,Math.min(data.grid-1,Math.floor(local.y/data.height*data.grid)));inspected=row*data.grid+col;inspectPoint();};
    svg.onkeydown=event=>{const offsets={ArrowRight:1,ArrowLeft:-1,ArrowDown:data.grid,ArrowUp:-data.grid};if(event.key in offsets){event.preventDefault();inspected=Math.max(0,Math.min(data.total-1,inspected+offsets[event.key]));inspectPoint();}};
  }
}
function inspectPoint(){
  if(!data)return;const details=pointDescription(inspected),[x,y]=data.points[inspected];$("point-inspector").replaceChildren();
  const position=document.createElement("b");position.textContent=details.position;$("point-inspector").append(position);const divider=document.createElement("span");divider.className="inspection-divider";divider.textContent="|";$("point-inspector").append(divider);$("point-inspector").append(document.createTextNode(`Dense: full decoding · FSD: ${details.reason}.`));
  for(const method of ["dense","fsd"]){const marker=$(method+"-map").querySelector(".point-highlight");marker.setAttribute("cx",x*data.width);marker.setAttribute("cy",y*data.height);}
}
function draw(){
  if(!data)return;const visibleSelection=new Set(data.waves.slice(0,wave).flat());if(wave===4)for(const index of data.guards)visibleSelection.add(index);const selectedCount=current>=3?data.selected.length:current===2?visibleSelection.size:0;
  $("dense-count").textContent=current>=2?`${count(data.total)} / ${count(data.total)}`:`${count(data.total)} points`;$("fsd-count").textContent=current>=2?`${count(selectedCount)} / ${count(data.total)}`:`${count(data.total)} points`;
  const radius=Math.min(data.width,data.height)/(data.grid*4.25);
  maps.dense.forEach(dot=>{dot.setAttribute("class","prompt-point"+(current>=2?" complete":""));dot.setAttribute("r",radius*(current>=2?1:.65));dot.dataset.state=current>=2?"complete":"pending";});
  maps.fsd.forEach((dot,index)=>{const isSelected=current>=3?data.selectedSet.has(index):visibleSelection.has(index),status=current<2?"pending":isSelected?(data.guardOnlySet.has(index)&&wave===4?"guard":"selected"):"skipped";dot.setAttribute("class","prompt-point"+(status==="pending"?"":" "+status));dot.setAttribute("r",radius*(status==="selected"?1.35:status==="guard"?1.55:.62));dot.dataset.state=status;});
  for(const method of ["dense","fsd"]){const overlay=data.overlays?.[method];$(method+"-map").querySelector("image").setAttribute("href",picture(current===4&&overlay?overlay:data.image));}
  const denseStage=[0,1,2,2,3][current],fsdStage=[0,1,2,3,3][current];for(const [method,index] of [["dense",denseStage],["fsd",fsdStage]])$(method+"-nodes").querySelectorAll(".node").forEach((node,i)=>{node.classList.toggle("active",i===index);node.classList.toggle("done",i<index);});
  $("dense-summary").textContent=current===4?`${count(data.total)} completed requests → ${data.masks.dense??"—"} masks after filters + NMS`:`${count(data.total)} points receive full native decoding.`;
  $("fsd-summary").textContent=current===4?`${count(data.selected.length)} completed requests → ${data.masks.fsd} masks after filters + NMS`:current===2?`Wave ${wave} / 4 · ${count(selectedCount)} retained${wave===4?` · ${data.guard_only.length} added by the local guard`:""}`:current>=3?`${count(data.selected.length)} completed · ${count(data.total-data.selected.length)} skip full decoding`:`Preview all ${count(data.total)} points before choosing which to complete.`;
  const descriptions=[
    ["1 / Encode once",`Both methods use the same image and ${data.grid} × ${data.grid} grid. Every dot marks an actual independent foreground prompt.`],
    ["2 / Keep all prompt factors",`Dense SAM repeats full decoding for all ${count(data.total)} prompts. FSD builds each prompt's factors while reusing image terms.`],
    ["3 / Select point positions",`Four selection waves retain ${count(data.nmsSet.size)} distinct points; the local guard adds ${data.guard_only.length} more. Purple / orange dots show the actual ${count(data.selected.length)} retained positions. Gray points skip full completion.`],
    ["4 / Complete the chosen points",`Dense SAM completes ${count(data.total)} requests. FSD completes ${count(data.selected.length)} requests with native suffixes. Skipping full decoding saves work at the gray positions.`],
    ["5 / Filter the completed masks",`The same quality / stability filters and NMS produce ${data.masks.dense??"—"} Dense SAM masks and ${data.masks.fsd} FSD masks. Completed prompt requests and final masks are different counts.`]
  ];$("step-title").textContent=descriptions[current][0];$("step-description").textContent=descriptions[current][1];document.querySelectorAll("[data-step]").forEach(button=>button.setAttribute("aria-pressed",String(Number(button.dataset.step)===current)));$("pause").textContent=playing?"Pause":"Play";inspectPoint();reportHeight();
}
function stop(){playing=false;clearTimeout(timer);draw();}
function next(){if(!playing)return;if(current===4){stop();return;}if(current===2&&wave<4){wave++;draw();timer=setTimeout(next,1100);return;}current++;if(current===2)wave=1;draw();timer=setTimeout(next,current===2?1100:2000);}
function play(){if(!data)return;playing=true;draw();clearTimeout(timer);timer=setTimeout(next,current===2?1100:2000);}
function apply(input){stop();data=validate(input);current=0;wave=0;inspected=data.selected[0]??0;buildMaps();$("grid-label").textContent=`${data.grid} × ${data.grid} · ${count(data.total)} points`;$("data-source").textContent=data.source==="live"?"Actual selection from this live run":"Recorded inference · actual point positions";$("footnote").textContent=`Actual ${data.grid} × ${data.grid} run: ${count(data.total)} sampled points → ${count(data.selected.length)} completed FSD requests → ${data.masks.fsd} masks. Animation illustrates the order; playback duration is unrelated to inference time.`;document.querySelectorAll("button").forEach(button=>button.disabled=false);draw();if(!matchMedia("(prefers-reduced-motion: reduce)").matches)play();}
$("replay").addEventListener("click",()=>{current=0;wave=0;play();});$("pause").addEventListener("click",()=>{if(playing)stop();else{if(current===4){current=0;wave=0;}play();}});document.querySelectorAll("[data-step]").forEach(button=>button.addEventListener("click",()=>{current=Number(button.dataset.step);wave=current>=2?4:0;stop();}));
addEventListener("message",event=>{if(event.source!==parent||event.data?.type!=="fsd-live-trace")return;try{apply(event.data.trace);}catch(error){$("data-source").textContent=error.message;}});document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
async function loadRecorded(){const response=await fetch(new URL("../every-comparison.json",location.href));if(!response.ok)throw new Error("Could not load measured selection");const pair=(await response.json()).examples.fruit,info=pair.results.fsd.info,selection=info.selection;if(!selection)throw new Error("This recorded run has no selection coordinates");apply({source:"recorded",grid:pair.grid,image_hw:info.image_hw,points:info.point_grid,selected:info.native_point_indices,waves:selection.wave_source_points,guards:selection.guard_source_points,guard_only:selection.guard_only_source_points,image:pair.image,overlays:{dense:pair.results.dense.overlay,fsd:pair.results.fsd.overlay},masks:{dense:pair.results.dense.info.masks,fsd:info.masks}});}
document.querySelectorAll("button").forEach(button=>button.disabled=true);const parameters=new URLSearchParams(location.search);if(parameters.get("live")==="1"||parameters.has("total"))$("data-source").textContent="Waiting for this live run's point coordinates…";else loadRecorded().catch(error=>{$("data-source").textContent=error.message;});reportHeight();

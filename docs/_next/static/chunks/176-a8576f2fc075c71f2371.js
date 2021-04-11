(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[176],{155:function(e,r,t){"use strict";t.d(r,{z:function(){return _}});var n=t(1180),a=t(917),i=t(5284),o=t(4686),c=t(5415),l=t(7294);function s(){return(s=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var u=(0,a.F4)({"0%":{transform:"rotate(0deg)"},"100%":{transform:"rotate(360deg)"}}),d=(0,n.forwardRef)(((e,r)=>{var t=(0,n.useStyleConfig)("Spinner",e),a=(0,i.Lr)(e),{label:d="Loading...",thickness:f="2px",speed:p="0.45s",color:h,emptyColor:v="transparent",className:m}=a,y=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(a,["label","thickness","speed","color","emptyColor","className"]),g=(0,o.cx)("chakra-spinner",m),_=s({display:"inline-block",borderColor:"currentColor",borderStyle:"solid",borderRadius:"99999px",borderWidth:f,borderBottomColor:v,borderLeftColor:v,color:h,animation:u+" "+p+" linear infinite"},t);return l.createElement(n.chakra.div,s({ref:r,__css:_,className:g},y),d&&l.createElement(c.TX,null,d))}));o.__DEV__&&(d.displayName="Spinner");var f=t(8500);function p(){return(p=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var[h,v]=(0,f.k)({strict:!1,name:"ButtonGroupContext"}),m=(0,n.forwardRef)(((e,r)=>{var{size:t,colorScheme:a,variant:i,className:c,spacing:s="0.5rem",isAttached:u,isDisabled:d}=e,f=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(e,["size","colorScheme","variant","className","spacing","isAttached","isDisabled"]),v=(0,o.cx)("chakra-button__group",c),m=l.useMemo((()=>({size:t,colorScheme:a,variant:i,isDisabled:d})),[t,a,i,d]),y={display:"inline-flex"};return y=p({},y,u?{"> *:first-of-type:not(:last-of-type)":{borderRightRadius:0},"> *:not(:first-of-type):not(:last-of-type)":{borderRadius:0},"> *:not(:first-of-type):last-of-type":{borderLeftRadius:0}}:{"& > *:not(style) ~ *:not(style)":{marginLeft:s}}),l.createElement(h,{value:m},l.createElement(n.chakra.div,p({ref:r,role:"group",__css:y,className:v},f)))}));function y(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}function g(){return(g=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}o.__DEV__&&(m.displayName="ButtonGroup");var _=(0,n.forwardRef)(((e,r)=>{var t,a=v(),c=(0,n.useStyleConfig)("Button",g({},a,e)),s=(0,i.Lr)(e),{isDisabled:u=(null==a?void 0:a.isDisabled),isLoading:d,isActive:f,isFullWidth:p,children:h,leftIcon:m,rightIcon:_,loadingText:x,iconSpacing:w="0.5rem",type:E="button",spinner:C,className:O,as:N}=s,S=y(s,["isDisabled","isLoading","isActive","isFullWidth","children","leftIcon","rightIcon","loadingText","iconSpacing","type","spinner","className","as"]),j=(0,o.mergeWith)({},null!=(t=null==c?void 0:c._focus)?t:{},{zIndex:1}),R=g({display:"inline-flex",appearance:"none",alignItems:"center",justifyContent:"center",transition:"all 250ms",userSelect:"none",position:"relative",whiteSpace:"nowrap",verticalAlign:"middle",outline:"none",width:p?"100%":"auto"},c,!!a&&{_focus:j});return l.createElement(n.chakra.button,g({disabled:u||d,ref:r,as:N,type:N?void 0:E,"data-active":(0,o.dataAttr)(f),"data-loading":(0,o.dataAttr)(d),__css:R,className:(0,o.cx)("chakra-button",O)},S),m&&!d&&l.createElement(b,{marginEnd:w},m),d&&l.createElement(k,{__css:{fontSize:"1em",lineHeight:"normal"},spacing:w,label:x},C),d?x||l.createElement(n.chakra.span,{opacity:0},h):h,_&&!d&&l.createElement(b,{marginStart:w},_))}));o.__DEV__&&(_.displayName="Button");var b=e=>{var{children:r,className:t}=e,a=y(e,["children","className"]),i=l.isValidElement(r)?l.cloneElement(r,{"aria-hidden":!0,focusable:!1}):r,c=(0,o.cx)("chakra-button__icon",t);return l.createElement(n.chakra.span,g({},a,{className:c}),i)};o.__DEV__&&(b.displayName="ButtonIcon");var k=e=>{var{label:r,spacing:t,children:a=l.createElement(d,{color:"currentColor",width:"1em",height:"1em"}),className:i,__css:c}=e,s=y(e,["label","spacing","children","className","__css"]),u=(0,o.cx)("chakra-button__spinner",i),f=g({display:"flex",alignItems:"center",position:r?"relative":"absolute",marginEnd:r?t:0},c);return l.createElement(n.chakra.div,g({className:u},s,{__css:f}),a)};o.__DEV__&&(k.displayName="ButtonSpinner")},2300:function(e,r,t){"use strict";var n=t(1180),a=t(4686),i=t(7294);function o(){return(o=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var c=e=>{var{type:r="checkbox",_hover:t,_invalid:a,_disabled:c,_focus:l,_checked:s,_child:u={opacity:0},_checkedAndChild:d={opacity:1},_checkedAndDisabled:f,_checkedAndFocus:p,_checkedAndHover:h,children:v}=e,m=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(e,["type","_hover","_invalid","_disabled","_focus","_checked","_child","_checkedAndChild","_checkedAndDisabled","_checkedAndFocus","_checkedAndHover","children"]),y="input[type="+r+"]:checked:disabled + &",g="input[type="+r+"]:checked:hover:not(:disabled) + &",_="input[type="+r+"]:checked:focus + &",b="input[type="+r+"]:disabled + &",k="input[type="+r+"]:focus + &",x="input[type="+r+"]:hover:not(:disabled):not(:checked) + &",w="input[type="+r+"]:checked + &, input[type="+r+"][aria-checked=mixed] + &",E="input[type="+r+"][aria-invalid=true] + &",C="& > *";return i.createElement(n.chakra.div,o({},m,{"aria-hidden":!0,__css:{display:"inline-flex",alignItems:"center",justifyContent:"center",transition:"all 120ms",flexShrink:0,[k]:l,[x]:t,[b]:c,[E]:a,[y]:f,[_]:p,[g]:h,[C]:u,[w]:o({},s,{[C]:d})}}),v)};a.__DEV__&&(c.displayName="ControlBox")},3652:function(e,r,t){"use strict";t.d(r,{lX:function(){return E}});var n=t(1180),a=t(5284),i=t(4686),o=t(7294),c=t(8327),l=!1,s=0,u=()=>++s;function d(e,r){var t=e||(l?u():null),[n,a]=o.useState(t);(0,c.G)((()=>{null===n&&a(u())}),[]),o.useEffect((()=>{!1===l&&(l=!0)}),[]);var i=null!=n?n.toString():void 0;return r?r+"-"+i:i}var f=t(639),p=t(1464),h=t(8500),v=t(2947);function m(){return(m=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}function y(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}var[g,_]=(0,h.k)({strict:!1,name:"FormControlContext"});var b=(0,n.forwardRef)(((e,r)=>{var t=(0,n.useMultiStyleConfig)("Form",e),c=function(e){var{id:r,isRequired:t,isInvalid:n,isDisabled:a,isReadOnly:i}=e,c=y(e,["id","isRequired","isInvalid","isDisabled","isReadOnly"]),l=d(),s=r||"field-"+l,u=s+"-label",h=s+"-feedback",g=s+"-helptext",[_,b]=o.useState(!1),[k,x]=o.useState(!1),[w,E]=(0,f.k)(),C=o.useCallback((function(e,r){return void 0===e&&(e={}),void 0===r&&(r=null),m({id:g},e,{ref:(0,v.l)(r,(e=>{e&&x(!0)}))})}),[g]),O=o.useCallback((function(e,r){return void 0===e&&(e={}),void 0===r&&(r=null),m({id:h},e,{ref:(0,v.l)(r,(e=>{e&&b(!0)})),"aria-live":"polite"})}),[h]),N=o.useCallback((function(e,r){return void 0===e&&(e={}),void 0===r&&(r=null),m({},e,c,{ref:r,role:"group"})}),[c]);return{isRequired:!!t,isInvalid:!!n,isReadOnly:!!i,isDisabled:!!a,isFocused:!!w,onFocus:(0,p.withFlushSync)(E.on),onBlur:E.off,hasFeedbackText:_,setHasFeedbackText:b,hasHelpText:k,setHasHelpText:x,id:s,labelId:u,feedbackId:h,helpTextId:g,htmlProps:c,getHelpTextProps:C,getErrorMessageProps:O,getRootProps:N}}((0,a.Lr)(e)),{getRootProps:l}=c,s=y(c,["getRootProps","htmlProps"]),u=(0,i.cx)("chakra-form-control",e.className),h=o.useMemo((()=>s),[s]);return o.createElement(g,{value:h},o.createElement(n.StylesProvider,{value:t},o.createElement(n.chakra.div,m({},l({},r),{className:u,__css:{width:"100%",position:"relative"}}))))}));i.__DEV__&&(b.displayName="FormControl");var k=(0,n.forwardRef)(((e,r)=>{var t=_(),a=(0,n.useStyles)(),c=(0,i.cx)("chakra-form__helper-text",e.className);return o.createElement(n.chakra.div,m({},null==t?void 0:t.getHelpTextProps(e,r),{__css:a.helperText,className:c}))}));function x(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}function w(){return(w=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}i.__DEV__&&(k.displayName="FormHelperText");var E=(0,n.forwardRef)(((e,r)=>{var t=(0,n.useStyleConfig)("FormLabel",e),c=(0,a.Lr)(e),{children:l,requiredIndicator:s=o.createElement(C,null)}=c,u=function(e){var r,t,n=_();return w({},e,{"data-focus":(0,i.dataAttr)(null==n?void 0:n.isFocused),"data-disabled":(0,i.dataAttr)(null==n?void 0:n.isDisabled),"data-invalid":(0,i.dataAttr)(null==n?void 0:n.isInvalid),"data-readonly":(0,i.dataAttr)(null==n?void 0:n.isReadOnly),id:null!=(r=e.id)?r:null==n?void 0:n.labelId,htmlFor:null!=(t=e.htmlFor)?t:null==n?void 0:n.id})}(x(c,["className","children","requiredIndicator"])),d=_();return o.createElement(n.chakra.label,w({ref:r,className:(0,i.cx)("chakra-form__label",c.className),__css:w({display:"block",textAlign:"start"},t)},u),l,null!=d&&d.isRequired?s:null)}));i.__DEV__&&(E.displayName="FormLabel");var C=(0,n.forwardRef)(((e,r)=>{var{children:t,className:a}=e,c=x(e,["children","className"]),l=_(),s=(0,n.useStyles)();if(null==l||!l.isRequired)return null;var u=(0,i.cx)("chakra-form__required-indicator",a);return o.createElement(n.chakra.span,w({role:"presentation","aria-hidden":!0,ref:r},c,{__css:s.requiredIndicator,className:u}),t||"*")}));i.__DEV__&&(C.displayName="RequiredIndicator")},639:function(e,r,t){"use strict";t.d(r,{k:function(){return a}});var n=t(7294);function a(e){void 0===e&&(e=!1);var[r,t]=(0,n.useState)(e);return[r,{on:(0,n.useCallback)((()=>{t(!0)}),[]),off:(0,n.useCallback)((()=>{t(!1)}),[]),toggle:(0,n.useCallback)((()=>{t((e=>!e))}),[])}]}},8327:function(e,r,t){"use strict";t.d(r,{G:function(){return a}});var n=t(7294),a=t(4686).isBrowser?n.useLayoutEffect:n.useEffect},8017:function(e,r,t){"use strict";t.d(r,{xu:function(){return l}});var n=t(1180),a=t(4686),i=t(7294);function o(){return(o=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}function c(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}var l=(0,n.chakra)("div");a.__DEV__&&(l.displayName="Box");var s=(0,n.forwardRef)(((e,r)=>{var{size:t,centerContent:n=!0}=e,a=c(e,["size","centerContent"]),s=n?{display:"flex",alignItems:"center",justifyContent:"center"}:{};return i.createElement(l,o({ref:r,boxSize:t,__css:o({},s,{flexShrink:0,flexGrow:0})},a))}));a.__DEV__&&(s.displayName="Square");var u=(0,n.forwardRef)(((e,r)=>{var{size:t}=e,n=c(e,["size"]);return i.createElement(s,o({size:t,ref:r,borderRadius:"9999px"},n))}));a.__DEV__&&(u.displayName="Circle")},7922:function(e,r,t){"use strict";t.d(r,{E:function(){return l}});var n=t(1180),a=t(5284),i=t(4686),o=t(7294);function c(){return(c=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var l=(0,n.forwardRef)(((e,r)=>{var t=(0,n.useStyleConfig)("Code",e),l=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}((0,a.Lr)(e),["className"]);return o.createElement(n.chakra.code,c({ref:r,className:(0,i.cx)("chakra-code",e.className)},l,{__css:c({display:"inline-block"},t)}))}));i.__DEV__&&(l.displayName="Code")},4096:function(e,r,t){"use strict";t.d(r,{k:function(){return c}});var n=t(1180),a=t(4686),i=t(7294);function o(){return(o=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var c=(0,n.forwardRef)(((e,r)=>{var{direction:t,align:a,justify:c,wrap:l,basis:s,grow:u,shrink:d}=e,f=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(e,["direction","align","justify","wrap","basis","grow","shrink"]),p={display:"flex",flexDirection:t,alignItems:a,justifyContent:c,flexWrap:l,flexBasis:s,flexGrow:u,flexShrink:d};return i.createElement(n.chakra.div,o({ref:r,__css:p},f))}));a.__DEV__&&(c.displayName="Flex")},336:function(e,r,t){"use strict";t.d(r,{X:function(){return l}});var n=t(1180),a=t(5284),i=t(4686),o=t(7294);function c(){return(c=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var l=(0,n.forwardRef)(((e,r)=>{var t=(0,n.useStyleConfig)("Heading",e),l=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}((0,a.Lr)(e),["className"]);return o.createElement(n.chakra.h2,c({ref:r,className:(0,i.cx)("chakra-heading",e.className)},l,{__css:t}))}));i.__DEV__&&(l.displayName="Heading")},9444:function(e,r,t){"use strict";t.d(r,{r:function(){return l}});var n=t(1180),a=t(5284),i=t(4686),o=t(7294);function c(){return(c=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var l=(0,n.forwardRef)(((e,r)=>{var t=(0,n.useStyleConfig)("Link",e),l=(0,a.Lr)(e),{className:s,isExternal:u}=l,d=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(l,["className","isExternal"]);return o.createElement(n.chakra.a,c({target:u?"_blank":void 0,rel:u?"noopener noreferrer":void 0,ref:r,className:(0,i.cx)("chakra-link",s)},d,{__css:t}))}));i.__DEV__&&(l.displayName="Link")},6034:function(e,r,t){"use strict";t.d(r,{Kq:function(){return u}});var n=t(1180),a=t(4686),i=t(1464),o=t(7294),c="& > *:not(style) ~ *:not(style)";function l(){return(l=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var s=e=>o.createElement(n.chakra.div,l({className:"chakra-stack__item"},e,{__css:l({display:"inline-block",flex:"0 0 auto",minWidth:0},e.__css)})),u=(0,n.forwardRef)(((e,r)=>{var{isInline:t,direction:u,align:d,justify:f,spacing:p="0.5rem",wrap:h,children:v,divider:m,className:y,shouldWrapChildren:g}=e,_=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(e,["isInline","direction","align","justify","spacing","wrap","children","divider","className","shouldWrapChildren"]),b=t?"row":null!=u?u:"column",k=o.useMemo((()=>function(e){var{spacing:r,direction:t}=e,n={column:{marginTop:r,marginEnd:0,marginBottom:0,marginStart:0},row:{marginTop:0,marginEnd:0,marginBottom:0,marginStart:r},"column-reverse":{marginTop:0,marginEnd:0,marginBottom:r,marginStart:0},"row-reverse":{marginTop:0,marginEnd:r,marginBottom:0,marginStart:0}};return{flexDirection:t,[c]:(0,a.mapResponsive)(t,(e=>n[e]))}}({direction:b,spacing:p})),[b,p]),x=o.useMemo((()=>function(e){var{spacing:r,direction:t}=e,n={column:{my:r,mx:0,borderLeftWidth:0,borderBottomWidth:"1px"},"column-reverse":{my:r,mx:0,borderLeftWidth:0,borderBottomWidth:"1px"},row:{mx:r,my:0,borderLeftWidth:"1px",borderBottomWidth:0},"row-reverse":{mx:r,my:0,borderLeftWidth:"1px",borderBottomWidth:0}};return{"&":(0,a.mapResponsive)(t,(e=>n[e]))}}({spacing:p,direction:b})),[p,b]),w=!!m,E=!g&&!w,C=(0,i.getValidChildren)(v),O=E?C:C.map(((e,r)=>{var t=r+1===C.length,n=g?o.createElement(s,{key:r},e):e;if(!w)return n;var a=t?null:o.cloneElement(m,{__css:x});return o.createElement(o.Fragment,{key:r},n,a)})),N=(0,a.cx)("chakra-stack",y);return o.createElement(n.chakra.div,l({ref:r,display:"flex",alignItems:d,justifyContent:f,flexDirection:k.flexDirection,flexWrap:h,className:N,__css:w?{}:{[c]:k[c]}},_),O)}));a.__DEV__&&(u.displayName="Stack");var d=(0,n.forwardRef)(((e,r)=>o.createElement(u,l({align:"center"},e,{direction:"row",ref:r}))));a.__DEV__&&(d.displayName="HStack");var f=(0,n.forwardRef)(((e,r)=>o.createElement(u,l({align:"center"},e,{direction:"column",ref:r}))));a.__DEV__&&(f.displayName="VStack")},4115:function(e,r,t){"use strict";t.d(r,{x:function(){return l}});var n=t(1180),a=t(5284),i=t(4686),o=t(7294);function c(){return(c=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var l=(0,n.forwardRef)(((e,r)=>{var t=(0,n.useStyleConfig)("Text",e),l=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}((0,a.Lr)(e),["className","align","decoration","casing"]),s=(0,i.filterUndefined)({textAlign:e.align,textDecoration:e.decoration,textTransform:e.casing});return o.createElement(n.chakra.p,c({ref:r,className:(0,i.cx)("chakra-text",e.className)},s,l,{__css:t}))}));i.__DEV__&&(l.displayName="Text")},3719:function(e,r,t){"use strict";t.d(r,{E:function(){return s}});var n=t(1180),a=t(9421),i=t(4686),o=t(7294);function c(){return(c=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}function l(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}var s=(0,n.forwardRef)(((e,r)=>{var{spacing:t="0.5rem",children:s,justify:d,direction:f,align:p,className:h,shouldWrapChildren:v}=e,m=l(e,["spacing","children","justify","direction","align","className","shouldWrapChildren"]),y=o.useMemo((()=>({"--chakra-wrap-spacing":e=>(0,i.mapResponsive)(t,(r=>(0,a.fr)("space",r)(e))),"--wrap-spacing":"calc(var(--chakra-wrap-spacing) / 2)",display:"flex",flexWrap:"wrap",justifyContent:d,alignItems:p,flexDirection:f,listStyleType:"none",padding:"0",margin:"calc(var(--wrap-spacing) * -1)","& > *:not(style)":{margin:"var(--wrap-spacing)"}})),[t,d,p,f]),g=v?o.Children.map(s,((e,r)=>o.createElement(u,{key:r},e))):s;return o.createElement(n.chakra.div,c({ref:r,className:(0,i.cx)("chakra-wrap",h)},m),o.createElement(n.chakra.ul,{className:"chakra-wrap__list",__css:y},g))}));i.__DEV__&&(s.displayName="Wrap");var u=(0,n.forwardRef)(((e,r)=>{var{className:t}=e,a=l(e,["className"]);return o.createElement(n.chakra.li,c({ref:r,__css:{display:"flex",alignItems:"flex-start"},className:(0,i.cx)("chakra-wrap__listitem",t)},a))}));i.__DEV__&&(u.displayName="WrapItem")},4255:function(e,r,t){"use strict";t.d(r,{W:function(){return a}});var n=t(7294);function a(e){return n.Children.toArray(e).filter((e=>n.isValidElement(e)))}},9561:function(e,r,t){"use strict";t.d(r,{y:function(){return i}});var n=t(4686),a=t(3935);function i(e){return r=>{var t=a.flushSync;(0,n.isFunction)(t)?t((()=>{e(r)})):e(r)}}},1464:function(e,r,t){"use strict";t.d(r,{withFlushSync:function(){return a.y},getValidChildren:function(){return i.W}});var n=t(8117);t.o(n,"getValidChildren")&&t.d(r,{getValidChildren:function(){return n.getValidChildren}}),t.o(n,"withFlushSync")&&t.d(r,{withFlushSync:function(){return n.withFlushSync}});var a=t(9561),i=t(4255)},2947:function(e,r,t){"use strict";t.d(r,{l:function(){return i}});var n=t(4686);function a(e,r){if(null!=e)if((0,n.isFunction)(e))e(r);else try{e.current=r}catch(t){throw new Error("Cannot assign value '"+r+"' to ref '"+e+"'")}}function i(){for(var e=arguments.length,r=new Array(e),t=0;t<e;t++)r[t]=arguments[t];return e=>{r.forEach((r=>a(r,e)))}}},8117:function(){},980:function(e,r,t){"use strict";t.d(r,{Switch:function(){return a.r}});t(2300);var n=t(4806);t.o(n,"Switch")&&t.d(r,{Switch:function(){return n.Switch}}),t.o(n,"useColorMode")&&t.d(r,{useColorMode:function(){return n.useColorMode}});var a=t(5513),i=t(1180);t.o(i,"useColorMode")&&t.d(r,{useColorMode:function(){return i.useColorMode}});var o=t(886);t.o(o,"useColorMode")&&t.d(r,{useColorMode:function(){return o.useColorMode}})},5513:function(e,r,t){"use strict";t.d(r,{r:function(){return g}});var n=t(7294),a=t(8327);function i(e){var r=n.useRef(e);return(0,a.G)((()=>{r.current=e})),n.useCallback((function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return null==r.current?void 0:r.current(...t)}),[])}var o=t(639);var c=t(4686),l=t(658),s=t(1464),u=t(2947),d=t(5415);function f(){return(f=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}function p(e){void 0===e&&(e={});var{defaultIsChecked:r,defaultChecked:t=r,isChecked:p,isFocusable:v,isDisabled:m,isReadOnly:y,isRequired:g,onChange:_,isIndeterminate:b,isInvalid:k,name:x,value:w,id:E,onBlur:C,onFocus:O,"aria-label":N,"aria-labelledby":S,"aria-invalid":j,"aria-describedby":R}=e,D=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(e,["defaultIsChecked","defaultChecked","isChecked","isFocusable","isDisabled","isReadOnly","isRequired","onChange","isIndeterminate","isInvalid","name","value","id","onBlur","onFocus","aria-label","aria-labelledby","aria-invalid","aria-describedby"]),I=i(_),P=i(C),L=i(O),[T,A]=(0,o.k)(),[M,V]=(0,o.k)(),[F,B]=(0,o.k)(),W=(0,n.useRef)(null),[H,q]=(0,n.useState)(!0),[z,G]=(0,n.useState)(!!t),[K,U]=function(e,r){var t=void 0!==e;return[t,t&&"undefined"!==typeof e?e:r]}(p,z);(0,l.ZK)({condition:!!r,message:'The "defaultIsChecked" prop has been deprecated and will be removed in a future version. Please use the "defaultChecked" prop instead, which mirrors default React checkbox behavior.'});var X=(0,n.useCallback)((e=>{y||m?e.preventDefault():(K||G(U?e.target.checked:!!b||e.target.checked),null==I||I(e))}),[y,m,U,K,b,I]);(0,a.G)((()=>{W.current&&(W.current.indeterminate=Boolean(b))}),[b]);var Z=m&&!v,J=(0,n.useCallback)((e=>{" "===e.key&&B.on()}),[B]),Q=(0,n.useCallback)((e=>{" "===e.key&&B.off()}),[B]);(0,a.G)((()=>{W.current&&(W.current.checked!==U&&G(W.current.checked))}),[W.current]);var Y=(0,n.useCallback)((function(e,r){void 0===e&&(e={}),void 0===r&&(r=null);return f({},e,{ref:r,"data-active":(0,c.dataAttr)(F),"data-hover":(0,c.dataAttr)(M),"data-checked":(0,c.dataAttr)(U),"data-focus":(0,c.dataAttr)(T),"data-indeterminate":(0,c.dataAttr)(b),"data-disabled":(0,c.dataAttr)(m),"data-invalid":(0,c.dataAttr)(k),"data-readonly":(0,c.dataAttr)(y),"aria-hidden":!0,onMouseDown:(0,l.v0)(e.onMouseDown,(e=>{e.preventDefault(),B.on()})),onMouseUp:(0,l.v0)(e.onMouseUp,B.off),onMouseEnter:(0,l.v0)(e.onMouseEnter,V.on),onMouseLeave:(0,l.v0)(e.onMouseLeave,V.off)})}),[F,U,m,T,M,b,k,y,B,V.off,V.on]),$=(0,n.useCallback)((function(e,r){return void 0===e&&(e={}),void 0===r&&(r=null),f({},D,e,{ref:(0,u.l)(r,(e=>{e&&q("LABEL"===e.tagName)})),onClick:(0,l.v0)(e.onClick,(()=>{var e;H||(null==(e=W.current)||e.click(),(0,c.focus)(W.current,{nextTick:!0}))})),"data-disabled":(0,c.dataAttr)(m)})}),[D,m,H]),ee=(0,n.useCallback)((function(e,r){return void 0===e&&(e={}),void 0===r&&(r=null),f({},e,{ref:(0,u.l)(W,r),type:"checkbox",name:x,value:w,id:E,onChange:(0,l.v0)(e.onChange,X),onBlur:(0,l.v0)(e.onBlur,A.off,P),onFocus:(0,l.v0)(e.onFocus,(0,s.withFlushSync)(A.on),L),onKeyDown:(0,l.v0)(e.onKeyDown,J),onKeyUp:(0,l.v0)(e.onKeyUp,Q),required:g,checked:U,disabled:Z,readOnly:y,"aria-label":N,"aria-labelledby":S,"aria-invalid":j?Boolean(j):k,"aria-describedby":R,"aria-disabled":m,style:d.NL})}),[x,w,E,X,A.off,A.on,P,L,J,Q,g,U,Z,y,N,S,j,k,R,m]),re=(0,n.useCallback)((function(e,r){return void 0===e&&(e={}),void 0===r&&(r=null),f({},e,{ref:r,onMouseDown:(0,l.v0)(e.onMouseDown,h),onTouchStart:(0,l.v0)(e.onTouchStart,h),"data-disabled":(0,c.dataAttr)(m),"data-checked":(0,c.dataAttr)(U),"data-invalid":(0,c.dataAttr)(k)})}),[U,m,k]);return{state:{isInvalid:k,isFocused:T,isChecked:U,isActive:F,isHovered:M,isIndeterminate:b,isDisabled:m,isReadOnly:y,isRequired:g},getRootProps:$,getCheckboxProps:Y,getInputProps:ee,getLabelProps:re,htmlProps:D}}function h(e){e.preventDefault(),e.stopPropagation()}var v=t(1180),m=t(5284);function y(){return(y=Object.assign||function(e){for(var r=1;r<arguments.length;r++){var t=arguments[r];for(var n in t)Object.prototype.hasOwnProperty.call(t,n)&&(e[n]=t[n])}return e}).apply(this,arguments)}var g=(0,v.forwardRef)(((e,r)=>{var t=(0,v.useMultiStyleConfig)("Switch",e),a=(0,m.Lr)(e),{spacing:i="0.5rem",children:o}=a,l=function(e,r){if(null==e)return{};var t,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(a[t]=e[t]);return a}(a,["spacing","children"]),{state:s,getInputProps:u,getCheckboxProps:d,getRootProps:f,getLabelProps:h}=p(l),g=n.useMemo((()=>y({display:"inline-block",verticalAlign:"middle",lineHeight:"normal"},t.container)),[t.container]),_=n.useMemo((()=>y({display:"inline-flex",flexShrink:0,justifyContent:"flex-start",boxSizing:"content-box",cursor:"pointer"},t.track)),[t.track]),b=n.useMemo((()=>y({userSelect:"none",marginStart:i},t.label)),[i,t.label]);return n.createElement(v.chakra.label,y({},f(),{className:(0,c.cx)("chakra-switch",e.className),__css:g}),n.createElement("input",y({className:"chakra-switch__input"},u({},r))),n.createElement(v.chakra.span,y({},d(),{className:"chakra-switch__track",__css:_}),n.createElement(v.chakra.span,{__css:t.thumb,className:"chakra-switch__thumb","data-checked":(0,c.dataAttr)(s.isChecked),"data-hover":(0,c.dataAttr)(s.isHovered)})),o&&n.createElement(v.chakra.span,y({className:"chakra-switch__label"},h(),{__css:b}),o))}));c.__DEV__&&(g.displayName="Switch")},886:function(e,r,t){"use strict";var n=t(7657);t.o(n,"useColorMode")&&t.d(r,{useColorMode:function(){return n.useColorMode}})},7657:function(){},5415:function(e,r,t){"use strict";t.d(r,{NL:function(){return i},TX:function(){return o}});var n=t(1180),a=t(4686),i={border:"0px",clip:"rect(0px, 0px, 0px, 0px)",height:"1px",width:"1px",margin:"-1px",padding:"0px",overflow:"hidden",whiteSpace:"nowrap",position:"absolute"},o=(0,n.chakra)("span",{baseStyle:i});a.__DEV__&&(o.displayName="VisuallyHidden");var c=(0,n.chakra)("input",{baseStyle:i});a.__DEV__&&(c.displayName="VisuallyHiddenInput")},7544:function(e,r,t){e.exports=t(6381)},6071:function(e,r,t){"use strict";var n=t(3848),a=t(9448);r.default=void 0;var i=a(t(7294)),o=t(1689),c=t(2441),l=t(5749),s={};function u(e,r,t,n){if(e&&(0,o.isLocalURL)(r)){e.prefetch(r,t,n).catch((function(e){0}));var a=n&&"undefined"!==typeof n.locale?n.locale:e&&e.locale;s[r+"%"+t+(a?"%"+a:"")]=!0}}var d=function(e){var r=!1!==e.prefetch,t=(0,c.useRouter)(),a=t&&t.pathname||"/",d=i.default.useMemo((function(){var r=(0,o.resolveHref)(a,e.href,!0),t=n(r,2),i=t[0],c=t[1];return{href:i,as:e.as?(0,o.resolveHref)(a,e.as):c||i}}),[a,e.href,e.as]),f=d.href,p=d.as,h=e.children,v=e.replace,m=e.shallow,y=e.scroll,g=e.locale;"string"===typeof h&&(h=i.default.createElement("a",null,h));var _=i.Children.only(h),b=_&&"object"===typeof _&&_.ref,k=(0,l.useIntersection)({rootMargin:"200px"}),x=n(k,2),w=x[0],E=x[1],C=i.default.useCallback((function(e){w(e),b&&("function"===typeof b?b(e):"object"===typeof b&&(b.current=e))}),[b,w]);(0,i.useEffect)((function(){var e=E&&r&&(0,o.isLocalURL)(f),n="undefined"!==typeof g?g:t&&t.locale,a=s[f+"%"+p+(n?"%"+n:"")];e&&!a&&u(t,f,p,{locale:n})}),[p,f,E,g,r,t]);var O={ref:C,onClick:function(e){_.props&&"function"===typeof _.props.onClick&&_.props.onClick(e),e.defaultPrevented||function(e,r,t,n,a,i,c,l){("A"!==e.currentTarget.nodeName||!function(e){var r=e.currentTarget.target;return r&&"_self"!==r||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey||e.nativeEvent&&2===e.nativeEvent.which}(e)&&(0,o.isLocalURL)(t))&&(e.preventDefault(),null==c&&(c=n.indexOf("#")<0),r[a?"replace":"push"](t,n,{shallow:i,locale:l,scroll:c}))}(e,t,f,p,v,m,y,g)},onMouseEnter:function(e){(0,o.isLocalURL)(f)&&(_.props&&"function"===typeof _.props.onMouseEnter&&_.props.onMouseEnter(e),u(t,f,p,{priority:!0}))}};if(e.passHref||"a"===_.type&&!("href"in _.props)){var N="undefined"!==typeof g?g:t&&t.locale,S=t&&t.isLocaleDomain&&(0,o.getDomainLocale)(p,N,t&&t.locales,t&&t.domainLocales);O.href=S||(0,o.addBasePath)((0,o.addLocale)(p,N,t&&t.defaultLocale))}return i.default.cloneElement(_,O)};r.default=d},5749:function(e,r,t){"use strict";var n=t(3848);r.__esModule=!0,r.useIntersection=function(e){var r=e.rootMargin,t=e.disabled||!o,l=(0,a.useRef)(),s=(0,a.useState)(!1),u=n(s,2),d=u[0],f=u[1],p=(0,a.useCallback)((function(e){l.current&&(l.current(),l.current=void 0),t||d||e&&e.tagName&&(l.current=function(e,r,t){var n=function(e){var r=e.rootMargin||"",t=c.get(r);if(t)return t;var n=new Map,a=new IntersectionObserver((function(e){e.forEach((function(e){var r=n.get(e.target),t=e.isIntersecting||e.intersectionRatio>0;r&&t&&r(t)}))}),e);return c.set(r,t={id:r,observer:a,elements:n}),t}(t),a=n.id,i=n.observer,o=n.elements;return o.set(e,r),i.observe(e),function(){o.delete(e),i.unobserve(e),0===o.size&&(i.disconnect(),c.delete(a))}}(e,(function(e){return e&&f(e)}),{rootMargin:r}))}),[t,r,d]);return(0,a.useEffect)((function(){if(!o&&!d){var e=(0,i.requestIdleCallback)((function(){return f(!0)}));return function(){return(0,i.cancelIdleCallback)(e)}}}),[d]),[p,d]};var a=t(7294),i=t(8391),o="undefined"!==typeof IntersectionObserver;var c=new Map},6381:function(e,r,t){"use strict";var n=t(809),a=t(2553),i=t(2012),o=t(9807),c=t(7690),l=t(9828),s=t(8561);function u(e){var r=function(){if("undefined"===typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"===typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var t,n=l(e);if(r){var a=l(this).constructor;t=Reflect.construct(n,arguments,a)}else t=n.apply(this,arguments);return c(this,t)}}var d=t(2426);r.Container=function(e){0;return e.children};var f=d(t(7294)),p=t(3937);function h(e){return v.apply(this,arguments)}function v(){return(v=s(n.mark((function e(r){var t,a,i;return n.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=r.Component,a=r.ctx,e.next=3,(0,p.loadGetInitialProps)(t,a);case 3:return i=e.sent,e.abrupt("return",{pageProps:i});case 5:case"end":return e.stop()}}),e)})))).apply(this,arguments)}p.AppInitialProps,p.NextWebVitalsMetric;var m=function(e){o(t,e);var r=u(t);function t(){return a(this,t),r.apply(this,arguments)}return i(t,[{key:"componentDidCatch",value:function(e,r){throw e}},{key:"render",value:function(){var e=this.props,r=e.router,t=e.Component,n=e.pageProps,a=e.__N_SSG,i=e.__N_SSP;return f.default.createElement(t,Object.assign({},n,a||i?{}:{url:y(r)}))}}]),t}(f.default.Component);function y(e){var r=e.pathname,t=e.asPath,n=e.query;return{get query(){return n},get pathname(){return r},get asPath(){return t},back:function(){e.back()},push:function(r,t){return e.push(r,t)},pushTo:function(r,t){var n=t?r:"",a=t||r;return e.push(n,a)},replace:function(r,t){return e.replace(r,t)},replaceTo:function(r,t){var n=t?r:"",a=t||r;return e.replace(n,a)}}}m.origGetInitialProps=h,m.getInitialProps=h},1664:function(e,r,t){e.exports=t(6071)},4155:function(e){var r,t,n=e.exports={};function a(){throw new Error("setTimeout has not been defined")}function i(){throw new Error("clearTimeout has not been defined")}function o(e){if(r===setTimeout)return setTimeout(e,0);if((r===a||!r)&&setTimeout)return r=setTimeout,setTimeout(e,0);try{return r(e,0)}catch(t){try{return r.call(null,e,0)}catch(t){return r.call(this,e,0)}}}!function(){try{r="function"===typeof setTimeout?setTimeout:a}catch(e){r=a}try{t="function"===typeof clearTimeout?clearTimeout:i}catch(e){t=i}}();var c,l=[],s=!1,u=-1;function d(){s&&c&&(s=!1,c.length?l=c.concat(l):u=-1,l.length&&f())}function f(){if(!s){var e=o(d);s=!0;for(var r=l.length;r;){for(c=l,l=[];++u<r;)c&&c[u].run();u=-1,r=l.length}c=null,s=!1,function(e){if(t===clearTimeout)return clearTimeout(e);if((t===i||!t)&&clearTimeout)return t=clearTimeout,clearTimeout(e);try{t(e)}catch(r){try{return t.call(null,e)}catch(r){return t.call(this,e)}}}(e)}}function p(e,r){this.fun=e,this.array=r}function h(){}n.nextTick=function(e){var r=new Array(arguments.length-1);if(arguments.length>1)for(var t=1;t<arguments.length;t++)r[t-1]=arguments[t];l.push(new p(e,r)),1!==l.length||s||o(f)},p.prototype.run=function(){this.fun.apply(null,this.array)},n.title="browser",n.browser=!0,n.env={},n.argv=[],n.version="",n.versions={},n.on=h,n.addListener=h,n.once=h,n.off=h,n.removeListener=h,n.removeAllListeners=h,n.emit=h,n.prependListener=h,n.prependOnceListener=h,n.listeners=function(e){return[]},n.binding=function(e){throw new Error("process.binding is not supported")},n.cwd=function(){return"/"},n.chdir=function(e){throw new Error("process.chdir is not supported")},n.umask=function(){return 0}},4405:function(e,r,t){"use strict";t.d(r,{w_:function(){return s}});var n=t(7294),a={color:void 0,size:void 0,className:void 0,style:void 0,attr:void 0},i=n.createContext&&n.createContext(a),o=function(){return(o=Object.assign||function(e){for(var r,t=1,n=arguments.length;t<n;t++)for(var a in r=arguments[t])Object.prototype.hasOwnProperty.call(r,a)&&(e[a]=r[a]);return e}).apply(this,arguments)},c=function(e,r){var t={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&r.indexOf(n)<0&&(t[n]=e[n]);if(null!=e&&"function"===typeof Object.getOwnPropertySymbols){var a=0;for(n=Object.getOwnPropertySymbols(e);a<n.length;a++)r.indexOf(n[a])<0&&Object.prototype.propertyIsEnumerable.call(e,n[a])&&(t[n[a]]=e[n[a]])}return t};function l(e){return e&&e.map((function(e,r){return n.createElement(e.tag,o({key:r},e.attr),l(e.child))}))}function s(e){return function(r){return n.createElement(u,o({attr:o({},e.attr)},r),l(e.child))}}function u(e){var r=function(r){var t,a=e.attr,i=e.size,l=e.title,s=c(e,["attr","size","title"]),u=i||r.size||"1em";return r.className&&(t=r.className),e.className&&(t=(t?t+" ":"")+e.className),n.createElement("svg",o({stroke:"currentColor",fill:"currentColor",strokeWidth:"0"},r.attr,a,s,{className:t,style:o(o({color:e.color||r.color},r.style),e.style),height:u,width:u,xmlns:"http://www.w3.org/2000/svg"}),l&&n.createElement("title",null,l),e.children)};return void 0!==i?n.createElement(i.Consumer,null,(function(e){return r(e)})):r(a)}}}]);
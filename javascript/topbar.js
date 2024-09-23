const browser={
    device: function(){
           var u = navigator.userAgent;
           // console.log(navigator);
           return {
                is_mobile: !!u.match(/AppleWebKit.*Mobile.*/),
                is_pc: (u.indexOf('Macintosh') > -1 || u.indexOf('Windows NT') > -1),
		is_wx_mini: (u.indexOf('miniProgram') > -1),
            };
         }(),
    language: (navigator.browserLanguage || navigator.language).toLowerCase()
}

let webpath = 'file';

async function set_language_by_ui(newLanguage) {
    if (newLanguage === "En") {
	newLocale="cn"
    } else {
	newLocale="en"
    }
    set_language(newLocale);
}

async function set_language(newLocale) {
    if (newLocale !== locale_lang) { 
        const newTranslations = await fetchTranslationsFor(newLocale);
        locale_lang = newLocale;
        localization = newTranslations;
    }
    console.log("localization[Preview]:"+localization["Preview"])
    onUiUpdate(function(m) {
        m.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                processNode(node);
            });
        });
    });
    localizeWholePage();
}

async function fetchTranslationsFor(newLocale) {
    let time_ver = "t="+Date.now()+"."+Math.floor(Math.random() * 10000)
    const response = await fetch(`${webpath}/language/${newLocale}.json?${time_ver}`);
    return await response.json();
}


function set_theme_by_ui(theme) {
    const gradioURL = window.location.href;
    const urls = gradioURL.split('?');
    if (theme=="dark") new_theme="light";
    if (theme=="light") new_theme="dark";
    const params = new URLSearchParams(window.location.search);
    const url_params = Object.fromEntries(params);
    let url_lang = locale_lang;
    if (url_params["__lang"]!=null) {
        url_lang=url_params["__lang"];
    }
    if (url_params["__theme"]!=null) {
        url_theme=url_params["__theme"];
	if (url_theme == new_theme) 
	    return
	window.location.replace(urls[0]+"?__theme="+new_theme+"&__lang="+url_lang+"&t="+Date.now()+"."+Math.floor(Math.random() * 10000));
    }
}

function set_iframe_src(theme = 'default', lang = 'cn', url) {
    const urlParams = new URLSearchParams(window.location.search);
    const themeParam = urlParams.get('__theme') || theme;
    const langParam = urlParams.get('__lang') || lang;

    console.log("langParam:"+langParam)

    // 构建新的iframe URL
    const newIframeUrl = `${url}${url.includes('?') ? '&' : '?'}__theme=${themeParam}&__lang=${langParam}`;

    // 获取iframe元素并设置src属性
    const iframe = gradioApp().getElementById('instruction');
    if (iframe) {
        iframe.src = newIframeUrl;
    } 

}

function closeSysMsg() {
    gradioApp().getElementById("sys_msg").style.display = "none";
}

function showSysMsg(message, theme) {
    const sysmsg = gradioApp().getElementById("sys_msg");
    const sysmsgText = gradioApp().getElementById("sys_msg_text");
    sysmsgText.innerHTML = message;
    
    const update_f = gradioApp().getElementById("update_f");
    const update_s = gradioApp().getElementById("update_s");

    if (theme == 'light') {
        sysmsg.style.color = "var(--neutral-600)";
        sysmsg.style.backgroundColor = "var(--secondary-100)";
	update_f.style.color = 'var(--primary-500)';
	update_s.style.color = 'var(--primary-500)';
    }
    else {
        sysmsg.style.color = "var(--neutral-100)";
        sysmsg.style.backgroundColor = "var(--secondary-400)";
	update_f.style.color = 'var(--primary-300)';
        update_s.style.color = 'var(--primary-300)';
    }

    sysmsg.style.display = "block";
}

function initPresetPreviewOverlay() {
    let overlayVisible = false;
    const samplesPath = document.querySelector("meta[name='preset-samples-path']").getAttribute("content")
    const overlay = document.createElement('div');
    const tooltip = document.createElement('div');
    tooltip.className = 'preset-tooltip';
    overlay.appendChild(tooltip);
    overlay.id = 'presetPreviewOverlay';
    document.body.appendChild(overlay);
    
    document.addEventListener('mouseover', async function (e) {
        const label = e.target.closest('.bar_button');
        if (!label) return;
        label.removeEventListener("mouseout", onMouseLeave);
        label.addEventListener("mouseout", onMouseLeave);
        const originalText = label.getAttribute("data-original-text");
        let name = originalText || label.textContent;
	name = name.trim();
	if (name!=" " && name!='') {
	    let download = false;
	    if (name.endsWith('\u2B07')) {
    	   	name = name.slice(0, -1);
    		download = true;
	    }
	    const img = new Image();
            img.src = samplesPath.replace(
                "default",
                name.toLowerCase().replaceAll(" ", "_")
            ).replaceAll("\\", "\\\\");
            img.onerror = async () => {
                overlay.style.height = '54px';
		let text = "模型资源"
		text += await fetchPresetDataFor(name);
                if (download) text += ' '+'\u2B07'+"未就绪要下载";
		else text += ' '+"已准备好";
                tooltip.textContent = text;
            };
	    img.onload = async () => {
                overlay.style.height = '128px'; 
		let text = await fetchPresetDataFor(name);
                if (download) text += ' '+'\u2B07'+"要下载资源";
                tooltip.textContent = text;
		overlay.style.backgroundImage = `url("${samplesPath.replace(
                    "default",
                    name.toLowerCase().replaceAll(" ", "_")
                ).replaceAll("\\", "\\\\")}")`;
            };

	    overlayVisible = true;
	    overlay.style.opacity = "1";
	}
        function onMouseLeave() {
            overlayVisible = false;
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            label.removeEventListener("mouseout", onMouseLeave);
        }
    });
    document.addEventListener('mousemove', function (e) {
        if (!overlayVisible) return;
        overlay.style.left = `${e.clientX}px`;
        overlay.style.top = `${e.clientY}px`;
        overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
    });
}

async function fetchPresetDataFor(name) {
    let time_ver = "t="+Date.now()+"."+Math.floor(Math.random() * 10000);
    const response = await fetch(`${webpath}/presets/${name}.json?${time_ver}`);
    const data = await response.json();
    let pos = data.default_model.lastIndexOf('.');
    return data.default_model.substring(0,pos);
}

function setObserver() {
    const elements = gradioApp().querySelectorAll('div#token_counter');
    for (var i = 0; i < elements.length; i++) {
	if (elements[i].className.includes('block')) {
            tokenCounterBlock = elements[i];
        }
        if (elements[i].className.includes('prose')) {
	    tokenCounter = elements[i];
	}
    }
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.target == tokenCounter) {
                var divTextContent = tokenCounter.textContent;
                if (parseInt(divTextContent) > 77 ) {
                    tokenCounterBlock.style.backgroundColor = 'var(--primary-700)'; 
                } else {
                    tokenCounterBlock.style.backgroundColor = 'var(--secondary-400)'; 
                }
            }
        });
    });
    var config = { childList: true, characterData: true };
    observer.observe(tokenCounter, config);
}

document.addEventListener("DOMContentLoaded", function() {
    const sysmsg = document.createElement('div');
    sysmsg.id = "sys_msg";
    sysmsg.className = 'systemMsg gradio-container';
    sysmsg.style.display = "none";
    sysmsg.tabIndex = 0;

    const sysmsgBox = document.createElement('div');
    sysmsgBox.id = "sys_msg_box";
    sysmsgBox.className = 'systemMsgBox gradio-container';
    sysmsgBox.style.setProperty("overflow-x", "auto");
    sysmsgBox.style.setProperty("border", "1px");
    sysmsgBox.style.setProperty("scrollbar-width", "thin");
    sysmsg.appendChild(sysmsgBox);

    const sysmsgText = document.createElement('pre');
    sysmsgText.id = "sys_msg_text";
    sysmsgText.style.setProperty("margin", "5px 12px 12px 0px");
    sysmsgText.innerHTML = '<b id="update_f">[Fooocus最新更新]</b>:'+'<b id="update_s">[SimpleSDXL最新更新]</b>';
    sysmsgBox.appendChild(sysmsgText);

    const sysmsgClose = document.createElement('div');
    sysmsgClose.className = 'systemMsgClose gradio-container';
    sysmsgClose.onclick = closeSysMsg;
    sysmsg.append(sysmsgClose);

    const sysmsgCloseText = document.createElement('span');
    sysmsgCloseText.innerHTML = 'x';
    sysmsgCloseText.style.setProperty("cursor", "pointer");
    sysmsgCloseText.onclick = closeSysMsg;
    sysmsgClose.appendChild(sysmsgCloseText);

    const sysmsgHeadTarget = document.createElement('base');
    sysmsgHeadTarget.target = "_blank"
    document.getElementsByTagName("head")[0].appendChild(sysmsgHeadTarget);

    try {
        gradioApp().appendChild(sysmsg);
    } catch (e) {
        gradioApp().body.appendChild(sysmsg);
    }

    document.body.appendChild(sysmsg);
    initPresetPreviewOverlay();
    
});


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
let nickname = 'guest';
let task_class_name = 'Fooocus';

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
    const params = new URLSearchParams(window.location.search);
    const url_params = Object.fromEntries(params);
    let url_lang = locale_lang;
    if (url_params["__lang"]!=null) {
        url_lang=url_params["__lang"];
    }
    if (url_params["__theme"]!=null) {
        url_theme=url_params["__theme"];
	if (url_theme == theme) 
	    return
	window.location.replace(urls[0]+"?__theme="+theme+"&__lang="+url_lang+"&t="+Date.now()+"."+Math.floor(Math.random() * 10000));
    }
}

function set_iframe_src(theme = 'default', lang = 'cn', url) {
    const urlParams = new URLSearchParams(window.location.search);
    const themeParam = urlParams.get('__theme') || theme;
    const langParam = urlParams.get('__lang') || lang;

    console.log("langParam:"+langParam)
    const newIframeUrl = `${url}${url.includes('?') ? '&' : '?'}__theme=${themeParam}&__lang=${langParam}`;
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
	let text = label.textContent.trim();
        let name = originalText || text;
	name = name.trim();
	if (name!=" " && name!='' && text!='') {
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

function getCookie(name) {
    const cookies = document.cookie.split(';').map(cookie => cookie.trim());
    const cookie = cookies.find(cookie => cookie.startsWith(name + '='));
    if (cookie) {
        return cookie.split('=')[1];
    }
    return null;
}

function setCookie(name, value, days) {
    const expires = new Date();
    expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
    document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
}

function checkAndUpdateSession(sstoken, days) {
    if (sstoken) {
	setCookie('aitoken', `${sstoken}`, days);
    }
}

function setLinkColor(theme) {
    let linkColorHover;
    let linkColorVisited;
    if (theme === 'dark') {
	const darkElement = document.querySelector('.dark');
	if (darkElement) {
	    darkElement.style.setProperty('--link-text-color', 'var(--secondary-300)');
	    darkElement.style.setProperty('--link-text-color-hover', 'var(--secondary-200)');
	    darkElement.style.setProperty('--link-text-color-visited', 'var(--secondary-300)');
	}
    }
}


async function refresh_identity_qrcode(nickname, did, user_qrcode) {
    let Canvg;

    if (window.canvg && window.canvg.Canvg) {
      Canvg = window.canvg.Canvg;
    } else if (window.canvg && window.canvg.default) {
      Canvg = window.canvg.default;
    } else if (window.Canvg) {
      Canvg = window.Canvg;
    } else {
      console.error('Canvg not found');
    } 
    if (user_qrcode) {
	didstr = did.substr(0, 10);
        const svg = document.getElementById('qrcode');
	var svgText = `<text x="40" y="20" font-family="Arial, sans-serif" font-size="16" fill="blue">`;
        svgText = svgText + nickname + "(" + didstr + ")</text>";
        const svgContent = user_qrcode.replace('</svg>', `${svgText}</svg>`);
	const ctx = svg.getContext('2d');
        const v = await Canvg.from(ctx, svgContent);
        await v.render();
	const pngDataUrl = svg.toDataURL('image/png');
	const link = document.createElement('a');
        link.href = pngDataUrl;
        link.download = "SimpleAI_identity_" + didstr +".png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

function refresh_topbar_status_js(system_params) {
    console.log('sp.length:', Object.keys(system_params).length);
    const preset=system_params["__preset"];
    const theme=system_params["__theme"];
    const nav_name_list_str = system_params["__nav_name_list"];
    checkAndUpdateSession(system_params["sstoken"], 90);
    setLinkColor(theme);
    nickname = system_params["user_name"];
    task_class_name = system_params["task_class_name"];
    let nav_name_list = new Array();
    if (nav_name_list_str) { nav_name_list = nav_name_list_str.split(","); }
    for (let i=0;i<nav_name_list.length;i++) {
        let item_id = "bar"+i;
        let item_name = nav_name_list[i];
        let nav_item = gradioApp().getElementById(item_id);
        if (nav_item!=null) {
	    nav_item.setAttribute('data-original-text', item_name);
            if (item_name != preset) {
                if (theme == "light") {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.background= 'var(--neutral-100)';
                } else {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.background= 'var(--neutral-700)';
                }
            } else {
                if (theme == 'light') {
                    nav_item.style.color = 'var(--neutral-800)';
                    nav_item.style.background= 'var(--secondary-200)';
                } else {
                    nav_item.style.color = 'white';
                    nav_item.style.background= 'var(--secondary-400)';
                }
            }
        }
    }
    updatePresetStore(nav_name_list, system_params["user_role"], system_params["preset_store"], theme);
    
    const message=system_params["__message"];
    if (message!=null && message.length>60) {
        showSysMsg(message, theme);
    }
    let infobox=gradioApp().getElementById("infobox");
    if (infobox!=null) {
        let css = infobox.getAttribute("class")
        if (browser.device.is_mobile && css.indexOf("infobox_mobi")<0)
            infobox.setAttribute("class", css.replace("infobox", "infobox_mobi"));
    }
    webpath = system_params["__webpath"];
    const lang=system_params["__lang"];
    if (lang!=null) {
        set_language(lang);
    }
    let preset_url = system_params["__preset_url"];
    if (preset_url!=null) {
        set_iframe_src(theme,lang,preset_url);
    }
    const image_num_pages = system_params["__finished_nums_pages"]; 
    if (image_num_pages) {
	refresh_finished_images_catalog_label(image_num_pages);
    }
    refresh_identity_center_label(system_params["user_role"]);
    (async () => {
        try {
	    await Promise.all([
            	refresh_identity_qrcode(nickname, system_params["user_did"], system_params["user_qr"]),
            ]);
        } catch (error) {
            console.error('Error refreshing QR code:', error);
        }
    })();
    return
}

function updatePresetStore(nav_name_list, role, expand_flag, theme) {
    let nav_store = gradioApp().getElementById("bar_store");
    let mypresets_text = "MyPresets"
    if (expand_flag) {
        if (theme == "light") {
            nav_store.style.background= 'lightcyan';
        } else {
            nav_store.style.background= 'darkslategray';
        }
	mypresets_text = mypresets_text + "▶";
    } else {
        nav_store.style.background= '';
	mypresets_text = mypresets_text + "▼";
    }
    if (role=="guest") {
        nav_store.innerHTML = "Presets▼";
    } else {
        nav_store.innerHTML = mypresets_text;
    }
    const preset_store = gradioApp().querySelector('.preset_store');
    if (!preset_store) return;    
    
    if (theme == "light") {
        preset_store.style.backgroundColor= 'lightcyan';
    } else {
        preset_store.style.backgroundColor= 'darkslategray';
    }
    
    const allButtons = preset_store.querySelectorAll('button');
    allButtons.forEach(button => {
	const div = button.querySelector('div.gallery');
	const originalText = div.getAttribute("data-original-text");
        let text = div.textContent.trim();
        let item_name = originalText || text;
        item_name = item_name.trim();
	// console.log("updatePresetStore: otext="+originalText+", text="+text+", name="+item_name);
	if (item_name) {
            if (nav_name_list.includes(item_name)) {
                if (theme === 'light') {
		    button.style.background= 'var(--neutral-50)';
                } else {
		    button.style.background= 'var(--neutral-600)';
                }
            } else {
                button.style.background= '';
            }
	}
    });
}

function getRandomTip() {
  if (tips && tips.length > 0) {
    return tips[Math.floor(Math.random() * tips.length)];
  }
  return '';
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

    const canvas = document.createElement('canvas');
    canvas.width = 343;
    canvas.height = 343;
    canvas.id = "qrcode";
    canvas.style.display = "none";
    
    try {
        gradioApp().appendChild(sysmsg);
    } catch (e) {
        gradioApp().body.appendChild(sysmsg);
    }
    try {
        gradioApp().appendChild(canvas);
    } catch (e) {
        gradioApp().body.appendChild(canvas);
    }

    document.body.appendChild(sysmsg);
    document.body.appendChild(canvas);
    initPresetPreviewOverlay();
    
});


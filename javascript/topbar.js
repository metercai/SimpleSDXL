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

function set_iframe_src(theme, lang, url) {
    const url_params = Object.fromEntries(new URLSearchParams(window.location.search));
    var theme_ifr = url_params['__theme'] || theme;
    var lang_ifr = url_params['__lang'] || lang;
    var newIframeUrl = url;
    if (newIframeUrl.includes('?')) {
        newIframeUrl += '&';
    } else {
        newIframeUrl += '?';
    }
    newIframeUrl += "__theme="+theme_ifr+"&__lang="+lang_ifr;
    const instruction = gradioApp().getElementById('instruction');
    if (instruction!=null) {
	    instruction.src = newIframeUrl;
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

});


function get_topbar_current() {
    var current=["light","default"];
    var url=document.documentURI.split("?");
    if (url.length>1) {
        var args=url[1].split("&");
        for (var i=0;i<args.length;i++) {
            var arg=args[i].split("=");
            if (arg.length>1) {
                switch(arg[0]) {
                    case "__theme": current[0]=arg[1];break;
                    case "__preset": current[1]=arg[1];break;
                }
            }
        }
    }
    return current;
}

var current_topbar = get_topbar_current();
var c_theme_id = "theme_"+current_topbar[0];
var c_preset_id = "preset_"+current_topbar[1];
function nav_mOver(nav_item){
    if (nav_item.id != c_preset_id && nav_item.id != c_theme_id) {
        if (c_theme_id == 'theme_light') {
            nav_item.style.color = 'var(--neutral-800)';
            nav_item.style.backgroundColor= 'var(--neutral-300)';
        } else {
            nav_item.style.color = 'var(--neutral-100)';
            nav_item.style.backgroundColor= 'var(--neutral-400';
        }
    }
}
function nav_mOut(nav_item){
    if (nav_item.id != c_preset_id && nav_item.id != c_theme_id) {
        if (c_theme_id == 'theme_light') {
            nav_item.style.color = 'var(--neutral-400)';
            nav_item.style.backgroundColor= 'var(--neutral-50)';
        } else {
            nav_item.style.color = 'var(--neutral-400)';
            nav_item.style.backgroundColor= 'var(--neutral-950';
        }
    }
}

function update_topbar(preset_list_id, html_str) {
    preset_list = gradioApp().getElementById(preset_list_id);
    if (preset_list!=null) {
	preset_list.innerHTML = html_str
    }
}

function mark_position_for_topbar(nav_id_str,preset_name,theme_name) {
    var current_preset_id = "preset_"+preset_name;
    var current_theme_id = "theme_"+theme_name;
    c_theme_id = current_theme_id;
    c_preset_id = current_preset_id;
    var nav_id_list = new Array();
    nav_id_list = nav_id_str.split(",")
    for (var i=0;i<nav_id_list.length;i++) {
	nav_item=document.getElementById(nav_id_list[i]);
	if (nav_item.id != c_preset_id && nav_item.id != c_theme_id) {
            if (c_theme_id == 'theme_light') {
		if (nav_item.id  == 'update_f' || nav_item.id  == 'update_s') 
	            nav_item.style.color = 'var(--primary-500)'
		else {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.backgroundColor= 'var(--neutral-50)';
		}
            } else {
		if (nav_item.id  == 'update_f' || nav_item.id  == 'update_s')
                    nav_item.style.color = 'var(--primary-300)'
                else {
                    nav_item.style.color = 'var(--neutral-400)';
                    nav_item.style.backgroundColor= 'var(--neutral-950';
                }
	    }
        } else {
	    if (c_theme_id == 'theme_light') {
                nav_item.style.color = 'var(--neutral-800)';
                nav_item.style.backgroundColor= 'var(--secondary-200)';
            } else {
                nav_item.style.color = 'var(--neutral-50)';
                nav_item.style.backgroundColor= 'var(--secondary-400';
            }
	}
    }
}

function refresh_preset(newPreset) {
    var current = get_topbar_current();
    var host_path = document.baseURI.split("?")[0];
    var newurl = host_path+"?__theme="+current[0]+"&__preset="+newPreset+"&t="+Date.now()+"."+Math.floor(Math.random() * 10000);
    window.location.replace(newurl);
}
function refresh_theme(newTheme) {
    var current = get_topbar_current();
    var host_path = document.baseURI.split("?")[0];
    var newurl = host_path+"?__theme="+newTheme+"&__preset="+current[1]+"&t="+Date.now()+"."+Math.floor(Math.random() * 10000);
    window.location.replace(newurl)
}

function set_theme_preset(theme,preset) {
    var gradioURL = window.location.href;
    if (gradioURL.includes('__theme=') && gradioURL.includes('__preset=')) {
        return;
    }
    var URLs = gradioURL.split("?");
    var newURL = gradioURL;
    if (URLs.length==1) {
	newURL = gradioURL + '?';
    }
    if (!gradioURL.includes('__theme=')) {
        newURL += '&__theme=' + theme;
    }
    if (!gradioURL.includes('__preset=')) {
        newURL += '&__preset=' + preset;
    }
    window.location.replace(newURL);
}

function closeSysMsg() {
    gradioApp().getElementById("sys_msg").style.display = "none";
}

function showSysMsg(message) {
    const sysmsg = gradioApp().getElementById("sys_msg");
    const sysmsgText = gradioApp().getElementById("sys_msg_text");
    sysmsgText.innerHTML = message;
    if (c_theme_id == 'theme_light') {
        sysmsg.style.color = "var(--neutral-600)";
        sysmsg.style.backgroundColor = "var(--secondary-100)";
    }
    else {
        sysmsg.style.color = "var(--neutral-100)";
        sysmsg.style.backgroundColor = "var(--secondary-400)";
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


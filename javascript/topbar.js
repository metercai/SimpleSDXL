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

function mark_position_for_topbar(preset_name,theme_name,preset_pre) {
    var current_preset_id = "preset_"+preset_name;
    var current_theme_id = "theme_"+theme_name;
    var pre_preset_id = "preset_"+preset_pre;
    var p_obj=document.getElementById(current_preset_id);
    var p1_obj=document.getElementById(pre_preset_id);
    var t_obj=document.getElementById(current_theme_id);
    var bgcolor = "#ccddff";
    switch(current_theme_id) {
	case "theme_dark":  bgcolor = "#6495ed"; break;
    }
    if (p_obj!=null) {
	bgcolor1 = p_obj.style.backgroundColor;
        p_obj.style.backgroundColor=bgcolor;
	if (p1_obj!=null) {
	    p1_obj.style.backgroundColor=bgcolor1;
	}
    }
    if (t_obj!=null) {
	t_obj.style.backgroundColor=bgcolor;
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


function set_toolbox() {
    let toolbox = gradioApp().getElementById("toolbox");
    if (toolbox==null) {
        const gallery = gradioApp().getElementById("final_gallery");
        toolbox = document.createElement('div');
        toolbox.id = "toolbox";
	toolbox.className = "toolbox";
        toolbox.innerHTML = "<ul class=\"toolbox_item\"><li><a href=# >PromptInfo</a></li><li><a href=# >PromptImage</a></li><li><a href=# >ColumnStyle</a></li><li><a href=# >Delete</a></li>";
        gallery.appendChild(toolbox);
    }
}

var re_num = /^[.\d]+$/;

var original_lines = {};
var translated_lines = {};

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

function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

function textNodesUnder(el) {
    var n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while ((n = walk.nextNode())) a.push(n);
    return a;
}

function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;
    var parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return false;
    if (re_num.test(text)) return false;
    return true;
}

function getTranslation(text) {
    if (!text) return undefined;

    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }

    var tl = localization[text];
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }

    return tl;
}

function processTextNode(node) {
    var text = node.textContent.trim();

    if (!canBeTranslated(node, text)) return;

    var tl = getTranslation(text);
    if (tl !== undefined) {
        node.textContent = tl;
        if (text && node.parentElement) {
          node.parentElement.setAttribute("data-original-text", text);
        }
    }
}

function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }

    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) {
            node.title = tl;
        }
    }

    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) {
            node.placeholder = tl;
        }
    }

    textNodesUnder(node).forEach(function(node) {
	processTextNode(node);
    });
}

function refresh_style_localization() {
    processNode(document.querySelector('.style_selections'));
}

let styleGridOriginalElements = [];

function refresh_style_layout() {
    const container = document.querySelector(".style_grid");
    if (container) {

        if (styleGridOriginalElements.length === 0) {
            styleGridOriginalElements = [...container.querySelectorAll('.style_item')];
        }

        const sortedStyles = Array.from(document.querySelectorAll('.style_selections input:checked'))
            .map(cb => cb.nextElementSibling.textContent.trim());

        const searchBar = gradioApp().querySelector('textarea[data-testid="textbox"][placeholder*="搜索风格"], textarea[data-testid="textbox"][placeholder*="search styles"]');
        const searchText = (searchBar?.value?.trim() || '').toLowerCase();

        const selectedItems = sortedStyles.map(name =>
            styleGridOriginalElements.find(item => {
                const btn = item.querySelector('button');
                const btnText = btn?.textContent.trim();
                return btnText === name;
            })
        ).filter(Boolean);

        const visibleUnselected = styleGridOriginalElements.filter(item => {
            const btn = item.querySelector('button');

            const rawOriginalText = btn?.getAttribute('data-original-text') || '';
            const rawTranslatedText = btn?.textContent || '';

            const cleanOriginal = rawOriginalText.trim().toLowerCase();
            const cleanTranslated = rawTranslatedText.trim().toLowerCase();

            return !selectedItems.some(selected => selected === item) &&
                   (cleanOriginal.includes(searchText) ||
                    cleanTranslated.includes(searchText));
        });

        visibleUnselected.forEach(item => {
            const btnText = item.querySelector('button')?.textContent.trim();
        });

        const hiddenItems = styleGridOriginalElements.filter(item =>
            ![...selectedItems, ...visibleUnselected].includes(item)
        );

        const finalOrder = [...selectedItems, ...visibleUnselected, ...hiddenItems];

        container.innerHTML = '';
        finalOrder.forEach(item => container.appendChild(item));
    }
    setTimeout(() => gradioApp().dispatchEvent(new Event('resize')), 50);
}


function refresh_scene_localization() {
    processNode(document.querySelector('.scene_aspect_ratio_selections'));
}

function refresh_aspect_ratios_label(value) {
    var label = document.querySelector('#aspect_ratios_accordion div span');
    var translation = getTranslation("Aspect Ratios");
    if (typeof translation == "undefined") {
        translation = "Aspect Ratios";
    }
    value = value.split(",")[0]
    label.textContent = translation + " - " + htmlDecode(value);
}


function refresh_finished_images_catalog_label(value) {
    var label = document.querySelector('#finished_images_catalog div span');
    var translation = getTranslation("Finished Images Catalog");
    if (typeof translation == "undefined") {
        translation = "'s Finished Images Catalog";
    } else { translation = "的" + translation; }
    var translation_stat = getTranslation("total: xxx images and yyy pages");
    if (typeof translation_stat == "undefined") {
        translation_stat = "total: xxx images and yyy pages";
    }
    var xxx = value.split(",")[0];
    var yyy = value.split(",")[1];
    var finished_label = nickname + translation + " - " + htmlDecode(translation_stat.replace(/xxx/g, xxx).replace(/yyy/g, yyy));
    const randomTip = getRandomTip();
    if (randomTip && !browser.device.is_mobile) {
	var space_num = 56 - randomTip.length;
	const spaces = space_num > 0 ? '&nbsp;'.repeat(space_num) : '';
        label.innerHTML = finished_label + spaces + randomTip; 
    } else { label.innerHTML = finished_label; }
}

function refresh_identity_center_label(role, upstream) {
    let label = document.getElementById("identity_center");
    var translation = getTranslation("IdentityCenter");
    if (typeof translation == "undefined") {
        translation = "IdentityCenter";
    }
    var display_name = nickname;
    if (role=="admin") {
	display_name = nickname + ", admin";
    }
    let display_upstream = upstream ? "On" : "Off";
    label.textContent = translation + "(" + display_name + ") - " + display_upstream;
}

function refresh_input_image_tab_label() {
    var items = ["Image Prompt", "Upscale or Variation", "Inpaint or Outpaint"]
    var imageInputTabs = document.getElementById('image_input_tabs');
    var tabNav = imageInputTabs.querySelector('.tab-nav');
    var buttons = tabNav.querySelectorAll('button');
    buttons.forEach(function(button) {
	let itemText = button.getAttribute('data-original-text');
	if (items.includes(itemText)) {
	    var translation = getTranslation(itemText);
	    if (typeof translation == "undefined") {
                translation = itemText;
            }
	    let class_name = task_class_name !== "Fooocus" ? "." + task_class_name : "";
	    button.textContent = translation + class_name;
	    button.addEventListener('click', function() {
                button.textContent = translation + class_name;
            });
	}
    });
}

function localizeWholePage() {
    console.log("in localize")
    processNode(gradioApp());

    function elem(comp) {
        var elem_id = comp.props.elem_id ? comp.props.elem_id : "component-" + comp.id;
        return gradioApp().getElementById(elem_id);
    }

    for (var comp of window.gradio_config.components) {
        if (comp.props.webui_tooltip) {
            let e = elem(comp);

            let tl = e ? getTranslation(e.title) : undefined;
            if (tl !== undefined) {
                e.title = tl;
            }
        }
        if (comp.props.placeholder) {
            let e = elem(comp);
            let textbox = e ? e.querySelector('[placeholder]') : null;

            let tl = textbox ? getTranslation(textbox.placeholder) : undefined;
            if (tl !== undefined) {
                textbox.placeholder = tl;
            }
        }
    }
}

document.addEventListener("DOMContentLoaded", function() {
    if (!hasLocalization()) {
        return;
    }

    onUiUpdate(function(m) {
        m.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                processNode(node);
            });
        });
    });

    localizeWholePage();

    if (localization.rtl) { // if the language is from right to left,
        (new MutationObserver((mutations, observer) => { // wait for the style to load
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') {
                        observer.disconnect();

                        for (const x of node.sheet.rules) { // find all rtl media rules
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all'); // enable them
                            }
                        }
                    }
                });
            });
        })).observe(gradioApp(), {childList: true});
    }
});

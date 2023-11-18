#!/bin/sh

# webui.py
WEBUI="webui.py"
sed -i "s|f'Fooocus {fooocus_version.version} '|f'SimpleSDXL derived form Fooocus {fooocus_version.version} '|" $WEBUI
sed -i "s|label='Advanced', value=modules|label='高级选项', value=modules|" $WEBUI
sed -i "s|label='Negative Prompt', show_label=True, placeholder=\"Type prompt here.\"|label='Negative Prompt', show_label=True, placeholder=\"Type negative prompt here.\"|" $WEBUI
sed -i "s|label='Advanced', value=False|label='高级控图模式: ImagePrompt-元素提示，PyraCanny-物体形状，CPDS-整体构图，FaceSwap-换脸', value=False|" $WEBUI
sed -i "s|/file={get_current_html_path()}|{args_manager.args.webroot}/file={get_current_html_path()}|" $WEBUI

sed -i "s|share=args_manager.args.share|share=args_manager.args.share, root_path=args_manager.args.webroot|" $WEBUI

echo "patched ok!"

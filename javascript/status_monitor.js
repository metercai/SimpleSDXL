(function() {
    // ==================== 配置常量 ====================
    const CHECK_INTERVAL = 2000;         // 检测间隔毫秒
    const MAX_RETRY_COUNT = 3;

    // ==================== 状态管理 ====================
    let state = {
        initialTimestamp: null,
        currentQueueSize: 0,
        retryCount: 0,
        isConnected: false,
        currentTheme: 'light',
        isDragging: false,
        offsetX: 0,
        offsetY: 0
    };

    // ==================== 移动设备检测 ====================
    function isMobileDevice() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        // 检测常见的移动设备标识符
        return /android|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent.toLowerCase());
    }

    // 如果是移动设备，则直接退出，不执行后续逻辑
    if (isMobileDevice()) {
        console.log("当前设备为移动设备，状态监控组件不显示。");
        // return;
    }

    // ==================== DOM 元素创建 ====================
    const statusContainer = document.createElement('div');
    statusContainer.id = 'gradio-status-monitor';

    const statusIndicator = document.createElement('div');
    statusIndicator.className = 'status-indicator';

    const reconnectBtn = document.createElement('button');
    reconnectBtn.className = 'reconnect-btn';
    reconnectBtn.textContent = ' 重连';
    reconnectBtn.onclick = () => window.location.reload();

    // VRAM 占用百分比
    const vramUsage = document.createElement('div');
    vramUsage.className = 'vram-usage';

    // RAM 占用百分比
    const ramUsage = document.createElement('div');
    ramUsage.className = 'ram-usage';

    // 同时在线用户数
    const onlineUsersBadge = document.createElement('div');
    onlineUsersBadge.className = 'online-users-badge';

    // 同时在线节点数
    const onlineNodesBadge = document.createElement('div');
    onlineNodesBadge.className = 'online-nodes-badge';

    // ==================== 样式配置 ====================
    const style = document.createElement('style');
    style.textContent = `
        #gradio-status-monitor {
            position: fixed;
            top: 3px;
            right: 3px;
            z-index: 9999;
            font-family: Arial, sans-serif;
            background: transparent; /* 设置背景为透明 */
            pointer-events: auto; /* 修改为auto以支持拖拽 */
            cursor: grab; /* 显示可拖拽的手型光标 */
        }
        
        #gradio-status-monitor.dragging {
            cursor: grabbing; /* 拖拽时显示抓取状态的光标 */
            opacity: 0.8;
        }

        /* 亮色主题 */
        .status-indicator.light {
            padding: 4px 8px;
            border-radius: 3px;
            display: flex;
            flex-direction: column;
            align-items: flex-end; /* 右对齐 */
            font-size: 12px;
	    background: transparent; /* 设置背景为透明 */
            border: none;
            color: #333;
	    box-shadow: none; /* 移除阴影 */
        }
        .light .status-connected { color: #2c7a2c; }
        .light .status-disconnected { color: #c53030; }
        .light .status-exception { color: #d97706; }

        /* 暗色主题 */
        .status-indicator.dark {
            padding: 4px 8px;
            border-radius: 3px;
            display: flex;
            flex-direction: column;
            align-items: flex-end; /* 右对齐 */
            font-size: 12px;
	    background: transparent; /* 设置背景为透明 */
            border-color: #4a5568;
            color: #fff;
	    border: none;
	    box-shadow: none; /* 移除阴影 */
        }
        .dark .status-connected { color: #48bb78; }
        .dark .status-disconnected { color: #f56565; }
        .dark .status-exception { color: #ecc94b; }

        /* 第一行样式 */
        .queue-badge {
            margin-left: 6px;
            padding: 1px 6px;
            border-radius: 8px;
            font-size: 11px;
        }
        .light .queue-badge { background: #f0f0f0; }
        .dark .queue-badge { background: #2d3748; }

        /* 重连按钮样式 */
        .reconnect-btn {
	    margin-left: 8px;
            padding: 2px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
	    pointer-events: auto;
        }
        .light .reconnect-btn {
            border: 1px solid #c53030;
            background: #fff0f0;
            color: #c53030;
        }
        .dark .reconnect-btn {
            border: 1px solid #f56565;
            background: #2d1a1a;
            color: #f56565;
        }

        .vram-usage, .ram-usage, .online-users-badge, .online-nodes-badge {
            margin-top: 4px;
            font-size: 11px;
        }
	.light .vram-usage {
            background: #f0f0f0;
            padding: 1px 6px;
            border-radius: 8px;
            color: var(--neutral-700);
        }
        .light .ram-usage, .light .online-users-badge, .light .online-nodes-badge {
            background: #f0f0f0;
            padding: 1px 6px;
            border-radius: 8px;
	    color: var(--neutral-400);
        }
	.dark .vram-usage {
            background: #2d3748;
            padding: 1px 6px;
            border-radius: 8px;
            color: var(--neutral-300);
        }
        .dark .ram-usage, .dark .online-users-badge, .dark .online-nodes-badge {
            background: #2d3748;
            padding: 1px 6px;
            border-radius: 8px;
	    color: var(--neutral-500);
        }
    `;

    // ==================== 主题管理 ====================
    function detectTheme() {
        const params = new URLSearchParams(window.location.search);
        return params.get('__theme') || 'light';
    }

    function applyTheme() {
        statusIndicator.classList.remove('light', 'dark');
        statusIndicator.classList.add(state.currentTheme);
    }

    // ==================== 核心功能 ====================
    async function fetchAppStatus() {
        try {
            const response = await fetch('/run/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fn_index: 0,
                    data: []
                })
            });

            if (!response.ok) throw new Error('请求失败');

            const result = await response.json();
            const [timestampStr, queueSizeStr, vramTotalStr, ramTotalStr, vramUsedStr, ramUsedStr, onlineUsersStr, onlineDomainUsersStr, onlineNodesStr] = result.data[0].split(',');

            return {
                timestamp: parseFloat(timestampStr),
                queueSize: parseInt(queueSizeStr),
                ramUsed: parseInt(ramUsedStr),
                ramTotal: parseInt(ramTotalStr),
                vramUsed: parseInt(vramUsedStr),
                vramTotal: parseInt(vramTotalStr),
                onlineUsers: parseInt(onlineUsersStr),
		onlineDomainUsers: parseInt(onlineDomainUsersStr),
		onlineNodes: parseInt(onlineNodesStr),
            };
        } catch (error) {
            return null;
        }
    }

    function updateStatusUI(statusType, queueSize, ramUsed, ramTotal, vramUsed, vramTotal, onlineUsers, onlineDomainUsers, onlineNodes) {
        statusIndicator.innerHTML = '';
        const statusMap = {
            connected: { text: '连接', class: 'status-connected' },
            disconnected: { text: '断开', class: 'status-disconnected' },
            exception: { text: '异常', class: 'status-exception' }
        };
        const { text, class: statusClass } = statusMap[statusType];

        // 构建状态指示
        const statusEl = document.createElement('span');
        statusEl.className = statusClass;
        statusEl.innerHTML = `● ${text}`;

        // 队列数与状态显示在同一行
        const firstRow = document.createElement('div');
        firstRow.style.display = 'flex';
        firstRow.style.alignItems = 'center';
        firstRow.appendChild(statusEl);

        if (statusType === 'connected') {
	    const queueBadge = document.createElement('span');
            queueBadge.className = 'queue-badge';
            queueBadge.textContent = `队列: ${queueSize}`;
            firstRow.appendChild(queueBadge);
	} else if (statusType === 'exception') {
	    firstRow.appendChild(reconnectBtn);
	} else {
            const retryText = document.createElement('span');
            retryText.textContent = ` (${state.retryCount * CHECK_INTERVAL / 1000}s)`;
            firstRow.appendChild(retryText);
        }
	    
        statusIndicator.appendChild(firstRow);

        // 添加附加信息
        if (statusType === 'connected' && !isMobileDevice()) {
	    // 显示 VRAM 使用情况
            const vramPercent = ((vramUsed / vramTotal) * 100).toFixed(1);
            vramUsage.textContent = `显存: ${vramPercent}%`;
            statusIndicator.appendChild(vramUsage);

            // 显示 RAM 使用情况
            const ramPercent = ((ramUsed / ramTotal) * 100).toFixed(1);
            ramUsage.textContent = `内存: ${ramPercent}%`;
            statusIndicator.appendChild(ramUsage);

            // 显示在线用户数
            if (onlineDomainUsers===0) {
		onlineUsersBadge.textContent = `用户: ${onlineUsers}`;
	    } else {
		onlineUsersBadge.textContent = `用户: ${onlineUsers}/${onlineDomainUsers}`;
	    }
            statusIndicator.appendChild(onlineUsersBadge);

	    // 显示在线节点数
	    if (onlineNodes!=0) {
                onlineNodesBadge.textContent = `节点: ${onlineNodes}`;
                statusIndicator.appendChild(onlineNodesBadge);
	    }
        }
    }

    async function performHealthCheck() {
        const statusData = await fetchAppStatus();

        if (!statusData) {
            state.retryCount++;
            if (state.retryCount >= MAX_RETRY_COUNT) {
                updateStatusUI('disconnected');
            }
            return;
        }

        state.retryCount = 0;

        if (!state.initialTimestamp) {
            state.initialTimestamp = statusData.timestamp;
            state.isConnected = true;
        }

        if (statusData.timestamp === state.initialTimestamp) {
            updateStatusUI(
                'connected',
                statusData.queueSize,
                statusData.ramUsed,
                statusData.ramTotal,
                statusData.vramUsed,
                statusData.vramTotal,
                statusData.onlineUsers,
		statusData.onlineDomainUsers,
		statusData.onlineNodes,
            );
        } else {
            updateStatusUI('exception');
        }
    }

    // ==================== 浏览器检测 ====================
    function detectBrowser() {
        const userAgent = navigator.userAgent.toLowerCase();
        if (userAgent.indexOf('chrome') > -1) return 'chrome';
        if (userAgent.indexOf('safari') > -1 && userAgent.indexOf('chrome') === -1) return 'safari';
        if (userAgent.indexOf('firefox') > -1) return 'firefox';
        if (userAgent.indexOf('edge') > -1) return 'edge';
        return 'unknown';
    }

    // ==================== 拖拽功能 ====================
    function initDragFeature() {
        const browser = detectBrowser();
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        
        // 标准鼠标拖拽事件
        statusContainer.addEventListener('mousedown', startDrag);
        
        // 触摸设备支持
        statusContainer.addEventListener('touchstart', startTouchDrag, { passive: false });
        
        // 为 Chrome 和 Edge 在 Mac 上添加 Pointer Events 支持
        if ((browser === 'chrome' || browser === 'edge') && isMac) {
            statusContainer.addEventListener('pointerdown', startPointerDrag);
        }
        
        function startDrag(e) {
            // 只响应左键 (button === 0)
            if (e.button === 0) {
                e.preventDefault();
                state.isDragging = true;
                
                // 获取当前位置
                const rect = statusContainer.getBoundingClientRect();
                state.offsetX = e.clientX - rect.left;
                state.offsetY = e.clientY - rect.top;
                
                statusContainer.classList.add('dragging');
                
                // 添加临时事件监听器
                document.addEventListener('mousemove', doDrag);
                document.addEventListener('mouseup', stopDrag);
            }
        }
        
        function startPointerDrag(e) {
            // 只响应主指针（通常是左键或触控板点击）
            if (e.isPrimary && (e.pointerType === 'mouse' || e.pointerType === 'touch')) {
                e.preventDefault();
                state.isDragging = true;
                
                // 获取当前位置
                const rect = statusContainer.getBoundingClientRect();
                state.offsetX = e.clientX - rect.left;
                state.offsetY = e.clientY - rect.top;
                
                statusContainer.classList.add('dragging');
                
                // 添加临时事件监听器
                document.addEventListener('pointermove', doPointerDrag);
                document.addEventListener('pointerup', stopPointerDrag);
                document.addEventListener('pointercancel', stopPointerDrag);
            }
        }
        
        function startTouchDrag(e) {
            if (e.touches && e.touches.length === 1) {
                e.preventDefault();
                state.isDragging = true;
                
                // 获取当前位置
                const rect = statusContainer.getBoundingClientRect();
                state.offsetX = e.touches[0].clientX - rect.left;
                state.offsetY = e.touches[0].clientY - rect.top;
                
                statusContainer.classList.add('dragging');
                
                // 添加临时事件监听器
                document.addEventListener('touchmove', doTouchDrag, { passive: false });
                document.addEventListener('touchend', stopTouchDrag);
                document.addEventListener('touchcancel', stopTouchDrag);
            }
        }
        
        function doDrag(e) {
            if (state.isDragging) {
                e.preventDefault();
                moveElement(e.clientX, e.clientY);
            }
        }
        
        function doPointerDrag(e) {
            if (state.isDragging) {
                e.preventDefault();
                moveElement(e.clientX, e.clientY);
            }
        }
        
        function doTouchDrag(e) {
            if (state.isDragging && e.touches && e.touches.length === 1) {
                e.preventDefault(); // 阻止页面滚动
                moveElement(e.touches[0].clientX, e.touches[0].clientY);
            }
        }
        
        // 统一移动元素的函数，增加边界保护
        function moveElement(clientX, clientY) {
            // 计算新位置
            const newLeft = clientX - state.offsetX;
            const newTop = clientY - state.offsetY;
            
            // 获取元素实际尺寸
            const rect = statusContainer.getBoundingClientRect();
            const elementWidth = rect.width;
            const elementHeight = rect.height;
            
            // 确保不超出视口边界，并留出余量防止变形
            const safeMargin = 3; // 安全边距，防止元素变形
            const maxX = window.innerWidth - elementWidth - safeMargin;
            const maxY = window.innerHeight - elementHeight - safeMargin;
            
            statusContainer.style.left = `${Math.max(safeMargin, Math.min(maxX, newLeft))}px`;
            statusContainer.style.top = `${Math.max(safeMargin, Math.min(maxY, newTop))}px`;
            statusContainer.style.right = 'auto'; // 取消右侧定位
            statusContainer.style.bottom = 'auto'; // 取消底部定位
        }
        
        function stopDrag() {
            if (state.isDragging) {
                state.isDragging = false;
                statusContainer.classList.remove('dragging');
                
                // 移除临时事件监听器
                document.removeEventListener('mousemove', doDrag);
                document.removeEventListener('mouseup', stopDrag);
            }
        }
        
        function stopPointerDrag() {
            if (state.isDragging) {
                state.isDragging = false;
                statusContainer.classList.remove('dragging');
                
                // 移除临时事件监听器
                document.removeEventListener('pointermove', doPointerDrag);
                document.removeEventListener('pointerup', stopPointerDrag);
                document.removeEventListener('pointercancel', stopPointerDrag);
            }
        }
        
        function stopTouchDrag() {
            if (state.isDragging) {
                state.isDragging = false;
                statusContainer.classList.remove('dragging');
                
                // 移除临时事件监听器
                document.removeEventListener('touchmove', doTouchDrag);
                document.removeEventListener('touchend', stopTouchDrag);
                document.removeEventListener('touchcancel', stopTouchDrag);
            }
        }
    }

    // ==================== 初始化 ====================
    function initializeMonitor() {
        // 检测并应用主题
        state.currentTheme = detectTheme();
        applyTheme();

        // 注入样式
        document.head.appendChild(style);

        // 组装 DOM
        statusContainer.appendChild(statusIndicator);

        // 集成到 Gradio
        const gradioContainer = gradioApp();
        if (gradioContainer) {
            gradioContainer.appendChild(statusContainer);
            
            // 初始化拖拽功能
            initDragFeature();
        }

        // 启动检测
        setInterval(performHealthCheck, CHECK_INTERVAL);
        performHealthCheck();
    }

    // 启动监控
    if (document.readyState === 'complete') {
        initializeMonitor();
    } else {
        window.addEventListener('load', initializeMonitor);
    }
})();


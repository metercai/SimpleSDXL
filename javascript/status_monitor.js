(function() {
    // ==================== 配置常量 ====================
    const CHECK_INTERVAL = 1000;         // 检测间隔毫秒
    const MAX_RETRY_COUNT = 3;

    // ==================== 状态管理 ====================
    let state = {
        initialTimestamp: null,
        currentQueueSize: 0,
        retryCount: 0,
        isConnected: false,
        currentTheme: 'light'
    };

    // ==================== DOM 元素创建 ====================
    const statusContainer = document.createElement('div');
    statusContainer.id = 'gradio-status-monitor';
    
    const statusIndicator = document.createElement('div');
    statusIndicator.className = 'status-indicator';
    
    const queueBadge = document.createElement('span');
    queueBadge.className = 'queue-badge';
    
    const reconnectBtn = document.createElement('button');
    reconnectBtn.className = 'reconnect-btn';
    reconnectBtn.textContent = '重连';
    reconnectBtn.onclick = () => window.location.reload();

    // ==================== 样式配置 ====================
    const style = document.createElement('style');
    style.textContent = `
        #gradio-status-monitor {
            position: fixed;
            top: 3px;
            right: 3px;
            z-index: 9999;
            font-family: Arial, sans-serif;
        }

        /* 亮色主题 */
        .status-indicator.light {
            padding: 4px 8px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            font-size: 12px;
            background: rgba(255,255,255,0.95);
            border: 1px solid #ddd;
            color: #333;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .light .status-connected { color: #2c7a2c; }
        .light .status-disconnected { color: #c53030; }
        .light .status-exception { color: #d97706; }

        /* 暗色主题 */
        .status-indicator.dark {
	    padding: 4px 8px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            font-size: 12px;
            background: rgba(26,32,44,0.95);
            border-color: #4a5568;
            color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        .dark .status-connected { color: #48bb78; }
        .dark .status-disconnected { color: #f56565; }
        .dark .status-exception { color: #ecc94b; }

        .queue-badge {
            margin-left: 6px;
            padding: 1px 6px;
            border-radius: 8px;
            font-size: 11px;
        }
        .light .queue-badge { background: #f0f0f0; }
        .dark .queue-badge { background: #2d3748; }

        .reconnect-btn {
            margin-left: 8px;
            padding: 2px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
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
            const [timestampStr, queueSizeStr] = result.data[0].split(',');
            
            return {
                timestamp: parseFloat(timestampStr),
                queueSize: parseInt(queueSizeStr)
            };
        } catch (error) {
            return null;
        }
    }

    function updateStatusUI(statusType, queueSize) {
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
        statusIndicator.appendChild(statusEl);

        // 添加附加信息
        if (statusType === 'connected') {
            queueBadge.textContent = `队列: ${queueSize}`;
            statusIndicator.appendChild(queueBadge);
        } else if (statusType === 'exception') {
            statusIndicator.appendChild(reconnectBtn);
        } else {
            const retryText = document.createElement('span');
            retryText.textContent = ` (${state.retryCount*CHECK_INTERVAL/1000}s)`;
            statusIndicator.appendChild(retryText);
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
            updateStatusUI('connected', statusData.queueSize);
        } else {
            updateStatusUI('exception');
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

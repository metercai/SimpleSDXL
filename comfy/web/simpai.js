// 新建 extractParams.js 文件
document.addEventListener('DOMContentLoaded', () => {
  // 样式注入
  const style = document.createElement('style');
  style.textContent = `
    #paramExtractBtn {
      position: fixed;
      right: 10px;
      top: 50%;
      transform: translateY(-50%);
      z-index: 9999;
      padding: 6px 6px;
      background: var(--primary-bg);
      color: var(--fg-color);
      border: 1px solid #4e4e4e;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.2s;
    }
    #paramExtractBtn:hover {
      background: #444;
      color: #eee;
    }
    #paramPopup {
      position: fixed;
      right: calc(10px + 32px);
      top: 50%;
      transform: translateY(-50%);
      width: 600px;
      max-height: 80vh;
      background: #2e2e2e;
      border: 1px solid #4e4e4e;
      border-radius: 4px;
      padding: 20px;
      color: #aaa;
      overflow: auto;
      display: none;
      z-index: 9998;
      box-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    #paramPopup pre {
      margin: 0;
      white-space: nowrap;
      font-family: monospace;
      line-height: 1.3;
    }
    #paramPopup.loading::after {
      content: "加载中...";
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: #aaa;
    }
    #paramPopup.active {
      display: block;
    }
  `;
  document.head.appendChild(style);

  // 创建界面元素
  const btn = document.createElement('button');
  btn.id = 'paramExtractBtn';
  btn.textContent = '参数提取';
  document.body.appendChild(btn);

  const popup = document.createElement('div');
  popup.id = 'paramPopup';
  const pre = document.createElement('pre');
  popup.appendChild(pre);
  document.body.appendChild(popup);

  // 状态管理
  let isOpen = false;
  let isLoading = false;

  function formatMappingResult(data) {
    try {
      if (!data || typeof data !== 'object') {
        throw new Error('无效的响应格式');
      }

      let output = '';
      const sortedKeys = Object.keys(data).sort();
      const maxLength = Math.max(...sortedKeys.map(key => key.length));

      // 遍历每个键值对
      for (const key of sortedKeys) {
        const values = data[key];
	const padding = ' '.repeat(maxLength - key.length + 1);
	// 确保值是数组
        if (Array.isArray(values)) {
          values.forEach(value => {
            output += `${key}${padding}--> ${value}\n`; // 使用换行符
          });
        } else {
          output += `${key}${padding}--> ${values}\n`; // 处理非数组情况
        }
      }

      return output || '无有效数据';
    } catch (error) {
      console.error('数据格式化失败:', error);
      return `数据显示错误: ${error.message}`;
    }
  }
  
  async function fetchMappingData(data) {
    try {
      popup.classList.add('loading');
      const response = await fetch('/mapping', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) throw new Error(`HTTP错误! 状态码: ${response.status}`);
      
      const result = await response.json();
      
      // 使用新格式化方法
      pre.textContent = formatMappingResult(result);
      
      // 添加样式类保持格式
      pre.style.whiteSpace = 'pre-wrap';
      pre.style.fontFamily = 'monospace';
      
    } catch (error) {
      console.error('请求失败:', error);
      pre.textContent = `错误: ${error.message}`;
    } finally {
      popup.classList.remove('loading');
    }
  }

  // 按钮点击处理
  btn.addEventListener('click', async () => {
    if (isLoading) return;

    if (!isOpen) {
      try {
        isLoading = true;
        const p = await app.graphToPrompt();
        await fetchMappingData(p.output);
        popup.style.display = 'block';
        popup.classList.add('active');
        isOpen = true;
      } catch (error) {
        console.error('参数提取失败:', error);
        pre.textContent = `错误: ${error.message}`;
      } finally {
        isLoading = false;
      }
    } else {
      popup.style.display = 'none';
      popup.classList.remove('active');
      isOpen = false;
    }
  });

  // 外部点击关闭
  document.addEventListener('click', (e) => {
    if (isOpen && !popup.contains(e.target) && e.target !== btn) {
      popup.style.display = 'none';
      popup.classList.remove('active');
      isOpen = false;
    }
  });
});

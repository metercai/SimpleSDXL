// tag_cart.js - Refactored with Fixed Layout, Preset Categories, and LAZY LOADING

// 声明 appRootInstance 在外部作用域
let appRootInstance;
let fullTagMap = new Map(); // [新增点 1] 声明一个全局的Map变量，用于快速查找标签

// 将所有核心逻辑封装到一个函数中
function initializeTagAssistantLogic() {
    console.log("initializeTagAssistantLogic started.");

    // ==================== 样式配置 ====================
    // [新增] 为加载指示器添加样式
    const newStyles = `
        .loading-indicator, .error-indicator {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            font-size: 1rem;
            color: #888;
        }
        /* [新增点 2] 为“未匹配标签”添加特殊样式 */
        .unmatched-tag {
            /* 使用CSS变量，使其能自适应亮/暗主题的背景色 */
            background-color: var(--neutral-100, #F3F4F6); /* 亮色主题下的背景色 */
            border: 1px dashed var(--neutral-400, #A3A3A3); /* 添加虚线边框以示区分 */
            color: var(--neutral-500, #737373); /* 文字颜色也变浅一些 */
        }
        [data-theme="dark"] .unmatched-tag {
            background-color: var(--neutral-800, #262626); /* 暗色主题下的背景色 */
        }
    `;
    const styleSheet = document.createElement("style");
    styleSheet.innerText = newStyles;
    document.head.appendChild(styleSheet);


    // --- 全局状态和常量 ---
    let webpath = '/file'; // gradio专用
    const localCsvUrl = `${webpath}/tags/danbooru_all.csv`;
    // [优化 1] 移除不再需要的远程URL
    // const giteeCsvUrl = '...';
    // const githubCsvUrl = '...';
    const customCsvUrl = `${webpath}/tags/custom_tags.csv`;
    
    const wildcardCnListUrl = `${webpath}/wildcards/cn_list.json`;
    const wildcardCnWordsUrl = `${webpath}/wildcards/cn_words.json`;

    // [优化 2] 简化 determineCsvUrl 函数，只使用本地路径
    async function determineCsvUrl() {
        // 根据要求，优化为仅使用本地URL，不再检查远程备份。
        // 这避免了不必要的网络请求和潜在的CORS跨域问题。
        console.log("使用本地 CSV 文件路径。");
        return localCsvUrl;
    }

    const TAGS_PER_PAGE = 32; // [修改点 4] 修改这个数字来改变每页显示的按钮数量
    const FIXED_WEIGHT = 1.1; // [修改点 4] 修改这个数字来改变追加标签格式化的默认权重
    const CATEGORIES_PER_PAGE = 13; // [修改点 4] 修改这个数字来改变每页显示的按钮数量

    // --- 全局UI文本配置 (无变化) ---
    const uiTexts = {
        // ... (内容与您提供的版本完全相同，为节省篇幅已折叠)
        searchInputPlaceholder: { zh: '搜索标签 (英文, 中文, 别名, 分类...)', en: 'Search tags (English, Chinese, aliases, categories...)' },
        draggableHandleText: { zh: '标签助手 v1.1 - 拖拽此处可移动', en: 'TagCart v1.1 - Drag Here' },
        formatButtonLabels: { 'spaces': { zh: '下划线转空格', en: 'Space for _' }, 'full': { zh: '空格+权重', en: 'Space + Weight' } },
        actionButtonLabels: { 'append': { zh: '追加', en: 'Append' }, 'replace': { zh: '替换', en: 'Replace' } },
        targetButtonLabels: { 'positive': { zh: '正向', en: 'Positive' }, 'negative': { zh: '反向', en: 'Negative' } },
        buttonTitles: {
            resetSearch: { zh: '清空搜索并重置类别', en: 'Clear search & reset categories' },
            copy: { zh: '发送到提示词框', en: 'Send to Prompt' },
            nsfwFilter: { zh: 'NSFW 过滤', en: 'NSFW Filter' },
            clearAll: { zh: '清空已选', en: 'Clear All Selected' },
            toggleLanguage: { zh: '切换显示语言', en: 'Toggle Display Language' }
        },
        presetCustomCategories: { // [修改点 2] 在这里添加新的自定义分类，直接删除或注释掉您不想要的那一行。添加ID标记和中英文名称。
            // 'ID'是这个分类在代码内部的唯一标识符（ID）。您的CSV文件里的<一级分类>列就需要填写这个字符串。zh 和 en 字段是它在界面上显示的中文和英文名称。
            'kontext指令': { zh: 'kontext指令', en: 'kontext' },
            '人物': { zh: '人物', en: 'People' },
            '画质': { zh: '画质', en: 'Quality' },
            '反向': { zh: '反向', en: 'Negative' }
        },
        primaryCategoryNames: {
            'All': { zh: '全部', en: 'All' }, 'General': { zh: '通用', en: 'General' }, 'Artist': { zh: '画师', en: 'Artist' }, 'Copyright': { zh: '作品', en: 'Copyright' },
            'Character': { zh: '角色', en: 'Character' }, 'Meta': { zh: '元数据', en: 'Meta' }, 'Kontext': { zh: 'Kontext指令', en: 'Kontext' }, 
            'Wildcard': { zh: '通配符', en: 'Wildcard' },
            'Custom': { zh: '自定义', en: 'Custom' }
        },
        tagTitleDefaults: {
            en: { noTranslation: 'None', noAliases: 'None', noCustomCategory: 'None', noSecondaryCategory: 'None', unknownCategory: 'Unknown Category' },
            zh: { noTranslation: '无', noAliases: '无', noCustomCategory: '无', noSecondaryCategory: '无', unknownCategory: '未知类别' }
        }
    };
    
    // --- 全局状态变量 ---
    // [优化 3] 新增 isDataLoaded 状态标志
    let isDataLoaded = false;
    let allTags = [], filteredTags = [], selectedTags = [];
    let searchableWildcardTags = [];
    let currentPage = 1, debounceTimer;
    let isNsfwFilterActive = true, displayEnglishOnly = false;
    let activeFormat = 'spaces', activeAction = 'append', activeTarget = 'positive';
    
    let primaryCategories = [], secondaryCategories = new Set();
    let wildcardFilenames = {}, wildcardTranslations = {}, wildcardWordTranslations = {};
    let activePrimaryCategory = 'All', activeSecondaryCategory = null;
    let primaryCategoryPage = 1, secondaryCategoryPage = 1;

    // --- DOM 元素引用 ---
    let selectedTagsContainer, tagDisplayContainer, searchInput, resetSearchBtn, nsfwFilterBtn, clearAllBtn, copyBtn;
    let importBtn; // [新增] 在这里声明 importBtn
    let paginationContainer, toggleLanguageBtn, draggableContainer, draggableHandle, closeBtn;
    let formatBtnGroup, actionBtnGroup, targetBtnGroup;
    let primaryCategoryRow, secondaryCategoryRow;

    // --- 初始化函数 (无变化) ---
    function init() {
        console.log("init() started.");
        appRootInstance = document.createElement('div');
        appRootInstance.id = 'app-root';
        appRootInstance.className = 'w-[970px] flex flex-col p-4 space-y-2 overflow-hidden min-h-[500px]';
        draggableContainer = document.createElement('div');
        draggableContainer.id = 'draggable-container';
        draggableContainer.className = 'absolute flex flex-col space-y-2 p-4 bg-neutral-100 dark:bg-neutral-800 rounded-2xl shadow-lg';
        draggableContainer.style.display = 'none';
        draggableContainer.style.width = '970px';
        draggableContainer.style.minHeight = '500px';

        const headerContainer = document.createElement('div');
        headerContainer.className = 'flex justify-between items-center w-full flex-shrink-0';
        draggableHandle = document.createElement('div');
        draggableHandle.id = 'draggable-handle';
        draggableHandle.className = 'flex-grow cursor-grab';
        closeBtn = document.createElement('button');
        closeBtn.id = 'close-draggable-btn';
        closeBtn.className = 'btn p-1 rounded-md w-5 h-5 flex items-center justify-center flex-shrink-0';
        closeBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
        closeBtn.title = '关闭';
        headerContainer.appendChild(draggableHandle);
        headerContainer.appendChild(closeBtn);
        draggableContainer.appendChild(headerContainer);
        
        selectedTagsContainer = document.createElement('div');
        selectedTagsContainer.id = 'selected-tags-container';
        selectedTagsContainer.className = 'p-2 rounded-lg h-[125px] flex flex-wrap gap-2 content-start overflow-y-auto';
        draggableContainer.appendChild(selectedTagsContainer);

        const controlBar = document.createElement('div');
        controlBar.className = 'flex-shrink-0 flex items-center gap-3';
        const searchWrapper = document.createElement('div');
        searchWrapper.className = 'relative flex-grow';
        searchInput = document.createElement('input');
        searchInput.type = 'text'; searchInput.id = 'search-input'; searchInput.className = 'input-control w-full p-2 pl-4 rounded-lg h-10';
        resetSearchBtn = document.createElement('button');
        resetSearchBtn.id = 'reset-search-btn'; resetSearchBtn.className = 'absolute right-2 top-1/2 -translate-y-1/2 p-1'; resetSearchBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
        searchWrapper.appendChild(searchInput); searchWrapper.appendChild(resetSearchBtn);
        controlBar.appendChild(searchWrapper);
        
        // [新增点 3] 创建并添加“导入”按钮
        // [修改] 移除 const，直接为全局变量赋值
        // [修改] 替换为这个新版本
        importBtn = document.createElement('button');
        importBtn.id = 'import-btn';

        // [核心修改]
        // 1. 添加 'flex items-center gap-2' 使其成为flex容器，并让图标和文字垂直居中、有2个单位的间距。
        // 2. 调整内边距，左右padding(px-3)比上下(py-2)稍大，更适合带文字的按钮。
        importBtn.className = 'btn px-3 py-2 rounded-lg h-10 flex-shrink-0 flex items-center gap-2';

        // [核心修改]
        // 现在 innerHTML 同时包含图标和包裹在 <span> 中的文字
        importBtn.innerHTML = '<i class="fa-solid fa-file-import"></i> <span>读取</span>';
        importBtn.title = '从正面提示词框中读取文本，并拆分成标签'; // 添加悬停提示
        controlBar.appendChild(importBtn); // 将它添加到 controlBar

        copyBtn = document.createElement('button'); copyBtn.id = 'copy-btn'; copyBtn.className = 'btn p-2 rounded-lg h-10 flex-shrink-0'; copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>';
        nsfwFilterBtn = document.createElement('button'); nsfwFilterBtn.id = 'nsfw-filter-btn'; nsfwFilterBtn.className = 'btn p-2 rounded-lg h-10 w-10 flex-shrink-0 active'; nsfwFilterBtn.innerHTML = '<i class="fa-solid fa-eye-slash"></i>';
        toggleLanguageBtn = document.createElement('button'); toggleLanguageBtn.id = 'toggle-language-btn'; toggleLanguageBtn.className = 'btn p-2 rounded-lg h-10 w-10 flex-shrink-0'; toggleLanguageBtn.innerHTML = '<i class="fa-solid fa-language"></i>';
        clearAllBtn = document.createElement('button'); clearAllBtn.id = 'clear-all-btn'; clearAllBtn.className = 'btn p-2 rounded-lg h-10 w-10'; clearAllBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
        controlBar.appendChild(copyBtn); controlBar.appendChild(nsfwFilterBtn); controlBar.appendChild(toggleLanguageBtn); controlBar.appendChild(clearAllBtn);
        draggableContainer.appendChild(controlBar);

        primaryCategoryRow = createCategoryRowDOM('primary-category-container', true);
        draggableContainer.appendChild(primaryCategoryRow);
        secondaryCategoryRow = createCategoryRowDOM('secondary-category-container', false);
        draggableContainer.appendChild(secondaryCategoryRow);
        
        tagDisplayContainer = document.createElement('div');
        tagDisplayContainer.id = 'tag-display-container';
        tagDisplayContainer.className = 'flex-grow grid grid-cols-8 grid-rows-4 gap-[6px] overflow-hidden p-1';
        draggableContainer.appendChild(tagDisplayContainer);

        const bottomWrapper = document.createElement('div');
        bottomWrapper.className = 'flex-shrink-0 flex justify-between items-center w-full mt-1';
        const otherControlBar = document.createElement('div');
        otherControlBar.id = 'other-control-bar';
        otherControlBar.className = 'flex items-center gap-4';
        const createButtonGroup = (valueMap, defaultValue, groupClass) => {
            const groupContainer = document.createElement('div');
            groupContainer.className = `flex items-center gap-1 p-1 rounded-lg ${groupClass} custom-group-bg`;
            Object.keys(valueMap).forEach(value => {
                const btn = document.createElement('button');
                btn.className = 'btn px-3 py-1 text-sm rounded-md';
                btn.dataset.value = value;
                if (value === defaultValue) btn.classList.add('active');
                groupContainer.appendChild(btn);
            });
            return groupContainer;
        };
        formatBtnGroup = createButtonGroup(uiTexts.formatButtonLabels, activeFormat, 'format-group');
        actionBtnGroup = createButtonGroup(uiTexts.actionButtonLabels, activeAction, 'action-group');
        targetBtnGroup = createButtonGroup(uiTexts.targetButtonLabels, activeTarget, 'target-group');
        otherControlBar.appendChild(formatBtnGroup);
        otherControlBar.appendChild(actionBtnGroup);
        otherControlBar.appendChild(targetBtnGroup);
        bottomWrapper.appendChild(otherControlBar);
        paginationContainer = document.createElement('div');
        paginationContainer.id = 'pagination-container';
        paginationContainer.className = 'flex-shrink-0 flex justify-center items-center gap-1';
        bottomWrapper.appendChild(paginationContainer);
        draggableContainer.appendChild(bottomWrapper);
        
        appRootInstance.appendChild(draggableContainer);
        console.log("init() completed.");
    }

    // ... createCategoryRowDOM, positionDraggableContainer, setupEventListeners, etc. ...
    // ... 这些函数与您提供的版本完全相同，为节省篇幅已折叠 ...
    // ... 它们内部没有任何逻辑需要为懒加载而修改 ...
    function createCategoryRowDOM(id, isPrimary = false) {
        const row = document.createElement('div');
        row.id = id;
        row.className = 'category-row flex-shrink-0 flex items-center gap-2';
        
        if (isPrimary) {
            const allBtn = document.createElement('button');
            allBtn.id = 'fixed-all-btn';
            allBtn.className = 'btn px-3 py-1 text-sm rounded-md flex-shrink-0';
            allBtn.dataset.categoryName = 'All';
            row.appendChild(allBtn);
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'buttons-wrapper flex-grow';
        row.appendChild(wrapper);

        const paginationDiv = document.createElement('div');
        paginationDiv.className = 'flex items-center gap-1 flex-shrink-0';
        
        const prevBtn = document.createElement('button');
        prevBtn.className = 'btn p-1 rounded-md w-6 h-6 flex items-center justify-center';
        prevBtn.innerHTML = '<i class="fa-solid fa-chevron-left"></i>';
        paginationDiv.appendChild(prevBtn);

        const nextBtn = document.createElement('button');
        nextBtn.className = 'btn p-1 rounded-md w-6 h-6 flex items-center justify-center';
        nextBtn.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
        paginationDiv.appendChild(nextBtn);

        row.appendChild(paginationDiv);

        return row;
    }
    function positionDraggableContainer() { // 添加一个函数来设置浮动框 draggableContainer 的初始位置坐标
        const containerWidth = 970;
        const containerHeight = 500;
        let initialTop = 220; //从顶部开始计算，正数为向下偏移
        let initialLeft = ((window.innerWidth - containerWidth) / 2) - 260; // 从左侧开始计算，正数为向右偏移，负数为向左偏移
        if (initialTop < 10) initialTop = 10;
        if (initialLeft < 10) initialLeft = 10;
        draggableContainer.style.top = `${initialTop}px`;
        draggableContainer.style.left = `${initialLeft}px`;
        draggableContainer.style.transform = '';
    }

    function setupEventListeners() {
        searchInput.addEventListener('input', () => {
            const query = searchInput.value.trim();
            
            if (query && activePrimaryCategory !== 'All') {
                handlePrimaryCategorySelect('All');
            }
            
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                applyFiltersAndRender();
            }, 300);
        });
        primaryCategoryRow.querySelector('#fixed-all-btn')?.addEventListener('click', () => {
            handlePrimaryCategorySelect('All');
        });

        const setupPaginationListeners = (rowElement, pageState, renderFunc, getCategoryList) => {
            const paginationDiv = rowElement.querySelector('.flex.items-center.gap-1');
            const prevBtn = paginationDiv.querySelector('button:first-child');
            const nextBtn = paginationDiv.querySelector('button:last-child');

            prevBtn.addEventListener('click', () => {
                if (pageState.page > 1) {
                    pageState.page--;
                    renderFunc();
                }
            });
            nextBtn.addEventListener('click', () => {
                const totalPages = Math.ceil(getCategoryList().length / CATEGORIES_PER_PAGE);
                if (pageState.page < totalPages) {
                    pageState.page++;
                    renderFunc();
                }
            });
        };

        setupPaginationListeners(primaryCategoryRow, { get page() { return primaryCategoryPage; }, set page(val) { primaryCategoryPage = val; } }, renderPrimaryCategories, () => primaryCategories);
        setupPaginationListeners(secondaryCategoryRow, { get page() { return secondaryCategoryPage; }, set page(val) { secondaryCategoryPage = val; } }, renderSecondaryCategories, () => Array.from(secondaryCategories));
        
        closeBtn.addEventListener('click', () => { draggableContainer.style.display = 'none'; });
        resetSearchBtn.addEventListener('click', () => { searchInput.value = ''; handlePrimaryCategorySelect('All'); });
        nsfwFilterBtn.addEventListener('click', () => { isNsfwFilterActive = !isNsfwFilterActive; nsfwFilterBtn.classList.toggle('active', isNsfwFilterActive); applyFiltersAndRender(); });
        clearAllBtn.addEventListener('click', () => { selectedTags = []; renderSelectedTags(); renderTags(); });
        copyBtn.addEventListener('click', copyTagsToClipboard);
        const setupButtonGroupListener = (groupElement, stateUpdater) => {
            groupElement.addEventListener('click', (e) => {
                const clickedButton = e.target.closest('button');
                if (!clickedButton) return;
                const value = clickedButton.dataset.value;
                if (value) {
                    stateUpdater(value);
                    groupElement.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
                    clickedButton.classList.add('active');
                }
            });
        };


        // [新增点 7] 绑定导入按钮的点击事件
        // [修改] 不再使用 getElementById，直接使用我们已经保存的变量
        if (importBtn) {
            console.log("找到导入按钮变量，正在绑定点击事件...");
            importBtn.addEventListener('click', importFromPrompt);
        } else {
            // 这个错误理论上不会再发生了
            console.error("致命错误：importBtn 变量未被正确初始化！");
        }

        closeBtn.addEventListener('click', () => { draggableContainer.style.display = 'none'; });


        setupButtonGroupListener(formatBtnGroup, value => activeFormat = value);
        setupButtonGroupListener(actionBtnGroup, value => activeAction = value);
        setupButtonGroupListener(targetBtnGroup, value => activeTarget = value);
        toggleLanguageBtn.addEventListener('click', () => {
            displayEnglishOnly = !displayEnglishOnly;
            toggleLanguageBtn.classList.toggle('active', displayEnglishOnly);
            updateUIText(displayEnglishOnly ? 'en' : 'zh');
        });
        let isDraggingContainer = false, containerOffset = { x: 0, y: 0 };
        draggableHandle.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            isDraggingContainer = true;
            containerOffset = { x: e.clientX - draggableContainer.getBoundingClientRect().left, y: e.clientY - draggableContainer.getBoundingClientRect().top };
            draggableHandle.style.cursor = 'grabbing';
            e.preventDefault();
        });
        document.addEventListener('mousemove', (e) => {
            if (!isDraggingContainer) return;
            let newX = e.clientX - containerOffset.x;
            let newY = e.clientY - containerOffset.y;
            draggableContainer.style.left = `${newX}px`;
            draggableContainer.style.top = `${newY}px`;
        });
        document.addEventListener('mouseup', () => {
            isDraggingContainer = false;
            draggableHandle.style.cursor = 'grab';
        });
    }


    // --- 功能模块: 数据加载与解析 ---
    
    // [优化 4] 新增懒加载主函数
    async function triggerDataLoadAndDisplay() {
        // 如果数据已经加载过，或者正在加载中（isDataLoaded已为true），则直接返回
        if (isDataLoaded) {
            return;
        }
        // 立即设置状态为 true，防止用户快速重复点击导致多次加载
        isDataLoaded = true; 
        console.log("首次激活面板，开始懒加载数据...");

        // 在UI上显示加载提示
        tagDisplayContainer.innerHTML = '<div class="loading-indicator">正在加载标签数据...</div>';

        try {
            await loadAllData(); // 调用真正的数据加载函数
            
            // 数据加载成功后，执行首次渲染和UI更新
            renderPrimaryCategories();
            applyFiltersAndRender();
            updateUIText(displayEnglishOnly ? 'en' : 'zh');
            
            console.log("数据懒加载并渲染完成。");

        } catch (error) {
            console.error("懒加载数据失败:", error);
            // 显示错误信息
            tagDisplayContainer.innerHTML = '<div class="error-indicator">数据加载失败，请检查控制台信息。</div>';
        }
    }

    /**
     * [新增点 4] 从正面提示词框导入、解析并更新已选区
     */
    // [修改] 替换为这个具备智能分隔符检测功能的新版本
    async function importFromPrompt() {
        console.log("[导入流程开始]");

        const promptTextarea = document.querySelector('gradio-app #positive_prompt textarea');
        if (!promptTextarea) {
            console.error("[导入中断] 找不到文本框。");
            return;
        }

        const text = promptTextarea.value;
        console.log(`[步骤1] 获取到文本: "${text}"`);
        if (!text.trim()) {
            console.log("[导入中断] 文本框为空。");
            alert("提示：正面提示词输入框是空的。");
            return;
        }

        // [核心修改] 智能检测分隔符
        let potentialTags;
        if (text.includes(',')) {
            // 模式一：检测到逗号，使用逗号作为唯一分隔符
            console.log("[解析模式] 检测到逗号，使用逗号作为主要分隔符。");
            potentialTags = text.split(',');
        } else {
            // 模式二：未检测到逗号，假定是Danbooru风格，使用空格作为分隔符
            console.log("[解析模式] 未检测到逗号，使用空格作为分隔符。");
            // 使用正则表达式 \s+ 来分割一个或多个连续的空白字符（空格、换行等）
            // 这可以避免因多个空格导致数组中出现空字符串。
            potentialTags = text.split(/\s+/);
        }

        console.log(`[步骤2] 分割为 ${potentialTags.length} 个潜在标签:`, potentialTags);

        selectedTags = []; // 重置
        const newlySelectedTags = [];
        const addedTagNames = new Set();

        for (const rawTag of potentialTags) {
            const cleanedName = cleanTagName(rawTag);
            // console.log(`[步骤3] 处理 "${rawTag}" -> 清洗为 "${cleanedName}"`);
            if (!cleanedName || addedTagNames.has(cleanedName)) {
                continue;
            }
            addedTagNames.add(cleanedName);

            if (fullTagMap.has(cleanedName)) {
                const foundTag = fullTagMap.get(cleanedName);
                newlySelectedTags.push(foundTag);
                //console.log(`  -> 匹配成功！添加官方标签:`, foundTag);
            } else {
                const unmatchedTag = {
                    name: cleanedName,
                    // translation: "未匹配的标签", // 未匹配标签不需要翻译
                    category: -99,
                    isUnmatched: true,
                    count: 0,
                    aliases: '',
                    customCategory: '',
                    secondaryCategory: ''
                };
                newlySelectedTags.push(unmatchedTag);
                //console.log(`  -> 匹配失败。添加为未匹配标签:`, unmatchedTag);
            }
        }

        selectedTags = newlySelectedTags;
        console.log(`[步骤4] 构建完成，新的 selectedTags 数组 (${selectedTags.length}个):`, selectedTags);

        console.log("[步骤5] 准备刷新UI...");
        renderSelectedTags();
        renderTags();
        console.log("[导入流程结束]");
    }

    /**
     * [新增点 5] 清洗从提示词中提取的单个标签字符串的辅助函数
     * @param {string} rawTag - 从提示词中分割出的原始字符串
     * @returns {string} - 清理好的、可用作查找键的标签名
     */
    function cleanTagName(rawTag) {
        if (!rawTag) return '';
        let tag = rawTag.trim();
        
        // 移除可能存在的Lora或Lyco格式，例如 <lora:name:1.0> -> ""
        tag = tag.replace(/<l(ora|yco):.*?>/g, '').trim();
        if (!tag) return '';

        // 处理转义括号 \( \)，变回 ( )
        tag = tag.replace(/\\\(/g, '(').replace(/\\\)/g, ')');
        
        // 循环去除首尾的圆括号和方括号，以处理多重嵌套如 ((tag))
        while (tag.startsWith('(') && tag.endsWith(')')) {
            tag = tag.substring(1, tag.length - 1).trim();
        }
        while (tag.startsWith('[') && tag.endsWith(']')) {
            tag = tag.substring(1, tag.length - 1).trim();
        }

        // 移除权重，例如 "masterpiece:1.2" -> "masterpiece"
        tag = tag.split(':')[0].trim();
        
        // 将空格替换为下划线，以匹配Danbooru格式
        tag = tag.replace(/ /g, '_');
        
        return tag;
    }

    // loadAllData 函数本身逻辑不变，它仍然是加载所有数据的核心
    async function loadAllData() {
        try {
            const finalCsvUrl = await determineCsvUrl();
            
            const [danbooruTags, customTagsResult, _] = await Promise.all([
                loadCSV(finalCsvUrl),
                loadCustomTags(),
                loadWildcardData()
            ]);
            
            allTags = [...customTagsResult, ...allTags];

            // [新增点 6] 填充全量标签Map以优化导入搜索性能
            fullTagMap.clear();
            [...allTags, ...searchableWildcardTags].forEach(tag => {
                if (tag.name) {
                    fullTagMap.set(tag.name, tag);
                }
            });

            console.log(`所有数据加载和解析完成。Map已填充，包含 ${fullTagMap.size} 个唯一标签。`);
            processCategories();
        } catch (error) {
            console.error("加载所有数据时发生严重错误:", error);
            throw error;
        }
    }

    // ... loadWildcardData, loadCSV, loadCustomTags 等函数 ...
    // ... 这些函数与您提供的版本完全相同，为节省篇幅已折叠 ...
    // ... 它们内部没有任何逻辑需要为懒加载而修改 ...

    // [修改] 替换您JS文件中旧的 loadWildcardData 函数
    async function loadWildcardData() {
        try {
            console.log("正在使用官方 API 获取通配符文件列表...");
            // [新增] 1. 调用官方API获取真实的文件名列表
            const apiResponse = await fetch('/run/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fn_index: 1, // 根据文档，fn_index 为 1
                    data: []
                })
            });

            if (!apiResponse.ok) {
                throw new Error(`通配符API请求失败: ${apiResponse.status} ${apiResponse.statusText}`);
            }

            const result = await apiResponse.json();
            const filenamesFromApi = result.data[0]; // "artists,clothes,scenery"
            
            // 将API返回的逗号分隔字符串转换为数组，并过滤掉可能的空值
            const officialFileNames = filenamesFromApi.split(',')
                .map(name => name.trim())
                .filter(Boolean); // filter(Boolean) 会移除空字符串

            console.log("从API获取到的官方文件名列表:", officialFileNames);

            // [修改] 2. 仍然加载翻译文件，但仅作查询使用
            const fetchWithTimeout = (url, options = {}, timeout = 5000) => {
                return Promise.race([fetch(url, options), new Promise((_, reject) => setTimeout(() => reject(new Error('Request timed out')), timeout))]);
            };

            const [cnListRes, cnWordsRes] = await Promise.all([
                fetchWithTimeout(wildcardCnListUrl).catch(e => { console.warn('无法加载通配符列表翻译:', e.message); return { ok: false }; }),
                fetchWithTimeout(wildcardCnWordsUrl).catch(e => { console.warn('无法加载通配符词条翻译:', e.message); return { ok: false }; })
            ]);

            let rawTranslations = {};
            if (cnListRes.ok) {
                rawTranslations = await cnListRes.json(); // 形如 {"wildcards/artists": "艺术家", ...}
            }
            if (cnWordsRes.ok) { 
                wildcardWordTranslations = await cnWordsRes.json(); 
            }

            // [修改] 3. 基于官方列表构建 wildcardFilenames，确保准确性
            wildcardFilenames = {}; // 清空旧数据
            officialFileNames.forEach(filename => {
                const translationKey = `list/${filename}`;
                // 从翻译文件中查找翻译，如果找不到，就用文件名本身作为显示文本
                wildcardFilenames[filename] = rawTranslations[translationKey] || filename;
            });

            console.log("整理后的通配符分类:", wildcardFilenames);
            
            // 4. 基于官方列表加载所有通配符词条用于搜索 (这部分逻辑不变，但数据源更准确了)
            if (officialFileNames.length === 0) { 
                console.log("API返回的通配符文件列表为空，跳过加载。"); 
                return; 
            }

            const allWildcardPromises = officialFileNames.map(async (filename) => {
                try {
                    // 注意：这里拼接路径是根据您之前代码的逻辑，确保 webpath 和路径正确
                    const response = await fetch(`${webpath}/wildcards/${filename}.txt`);
                    if (!response.ok) return [];
                    const text = await response.text();
                    return text.split('\n').map(line => line.trim()).filter(line => line !== '');
                } catch (e) { 
                    console.warn(`加载通配符文件 ${filename}.txt 失败:`, e); 
                    return []; 
                }
            });

            const allLinesNested = await Promise.all(allWildcardPromises);
            const allLinesFlat = allLinesNested.flat();

            searchableWildcardTags = allLinesFlat.map(line => ({
                name: line,
                translation: wildcardWordTranslations[line] || '',
                category: -2,
                count: 0,
                isWildcard: true,
                aliases: '',
                customCategory: '通配符',
                secondaryCategory: ''
            }));
            
            console.log(`通配符数据处理完成，共加载 ${searchableWildcardTags.length} 个可搜索词条。`);
        } catch (err) {
            console.error('处理通配符数据时出错:', err);
            // 出错时清空，避免显示错误/过时的按钮
            wildcardFilenames = {};
            searchableWildcardTags = [];
        }
    }

    async function loadCSV(source) {
        console.log("loadCSV started for source:", source);
        allTags = [];
        try {
            const response = await fetch(source);
            if (!response.ok) {
                throw new Error(`网络响应不佳: ${response.status} ${response.statusText}`);
            }
            const csvData = await response.text();
            
            return new Promise((resolve, reject) => {
                Papa.parse(csvData, {
                    worker: true, 
                    header: false, 
                    encoding: "UTF-8", 
                    step: (row) => { 
                        const data = row.data;
                        if (data.length >= 3 && data[0]) {
                        allTags.push({
                            name: data[0].trim(), 
                            category: parseInt(data[1], 10), 
                            count: parseInt(data[2], 10), 
                            aliases: (data[3] || '').trim(), 
                            translation: (data[4] || '').trim(), 
                            customCategory: (data[5] || '').trim(),
                            secondaryCategory: (data[6] || '').trim(),
                            remarks: (data[7] || '').trim() // <-- 新增这一行
                        });
                        }
                    },
                    complete: () => { 
                        console.log("CSV parsing completed.");
                        resolve();
                    },
                    error: (err) => { 
                        console.error("解析CSV时出错:", err);
                        reject(err);
                    }
                });
            });
        } catch (err) {
            console.error(`加载或获取 CSV 源 (${source}) 时出错:`, err);
            return Promise.reject(err);
        }
    }
    async function loadCustomTags() {
        console.log("加载自定义标签 custom_tags.csv...");
        const customTags = [];
        try {
            const response = await fetch(customCsvUrl);
            if (!response.ok) {
                console.warn("custom_tags.csv 未找到或加载失败，跳过。");
                return []; 
            }
            const csvData = await response.text();
            
            return new Promise((resolve, reject) => {
                Papa.parse(csvData, {
                    header: false,
                    skipEmptyLines: true,
                    step: (row) => {
                        const data = row.data;
                        if (data && data[0]) {
                            customTags.push({
                                name: (data[0] || '').trim(), 
                                category: 9, 
                                count: Infinity, 
                                aliases: (data[2] || '').trim(), 
                                translation: (data[1] || '').trim(),
                                customCategory: '自定义',
                                secondaryCategory: '',
                                isCustom: true
                            });
                        }
                    },
                    complete: () => {
                        console.log(`自定义标签加载完成，共 ${customTags.length} 个。`);
                        resolve(customTags);
                    },
                    error: (err) => {
                        console.error("解析 custom_tags.csv 时出错:", err);
                        reject(err);
                    }
                });
            });
        } catch (err) {
            console.error("获取 custom_tags.csv 时出错:", err);
            return [];
        }
    }

    // --- 功能模块: 分类处理与渲染 ---
    // --- 功能模块: 核心交互逻辑 ---
    // --- 功能模块: UI渲染与更新 ---
    // ... processCategories, renderPrimaryCategories, handlePrimaryCategorySelect, etc. ...
    // ... 剩余所有功能函数与您提供的版本完全相同，为节省篇幅已折叠 ...
    // ... 它们内部没有任何逻辑需要为懒加载而修改 ...
    function processCategories() {
        // [修改点 1] 在这里增删系统分类按钮
        const standardCategories = [
            'General', 'Character', 'Copyright', 'Artist', 'Meta', 
            //'Kontext', 
            'Custom']; // <-- 添加 'Custom
        const customCategories = Object.keys(uiTexts.presetCustomCategories);
        
        let processedCategories = [...standardCategories, ...customCategories];

        if (Object.keys(wildcardFilenames).length > 0) {
            processedCategories.push('Wildcard');
        }
        const desiredOrder = [
        // [修改点 1] “一级分类按钮的期望顺序”在这里调整分类按钮的顺序，这里填写的分类名称必须与上面 processCategories 函数中生成的分类名称ID一致
            'Wildcard', 
            //'Kontext', 
            'General', 'Character', 'Copyright', 'Artist',
            'Meta', 'Custom', 'kontext指令', '人物', '画质', '反向'
        ];

        primaryCategories = processedCategories.sort((a, b) => {
            const indexA = desiredOrder.indexOf(a);
            const indexB = desiredOrder.indexOf(b);
            if (indexA !== -1 && indexB !== -1) return indexA - indexB;
            if (indexA !== -1) return -1;
            if (indexB !== -1) return 1;
            return a.localeCompare(b);
        });
    }
    
    function renderCategoryRow(rowElement, categories, currentPage, activeCategory, clickHandler, isPrimary = false) {
        const wrapper = rowElement.querySelector('.buttons-wrapper');
        wrapper.innerHTML = '';
        
        const categoriesForPagination = isPrimary ? categories.filter(c => c !== 'All') : Array.from(categories);
        const totalPages = Math.ceil(categoriesForPagination.length / CATEGORIES_PER_PAGE);
        
        const paginationDiv = rowElement.querySelector('.flex.items-center.gap-1');
        const prevBtn = paginationDiv.querySelector('button:first-child');
        const nextBtn = paginationDiv.querySelector('button:last-child');
        
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage >= totalPages || categoriesForPagination.length === 0;

        if (categories.length === 0) return;

        const startIndex = (currentPage - 1) * CATEGORIES_PER_PAGE;
        const endIndex = startIndex + CATEGORIES_PER_PAGE;
        const categoriesToRender = categoriesForPagination.slice(startIndex, endIndex);

        categoriesToRender.forEach(category => {
            const btn = document.createElement('button');
            btn.className = 'btn px-3 py-1 text-sm rounded-md flex-shrink-0';
            btn.dataset.categoryName = category;
            
            const lang = displayEnglishOnly ? 'en' : 'zh';
            let displayText = category;
            if (activePrimaryCategory === 'Wildcard' && !isPrimary) {
                 displayText = wildcardFilenames[category] || category;
                 if (lang === 'en') displayText = category;
            } else if (uiTexts.primaryCategoryNames[category]) {
                displayText = uiTexts.primaryCategoryNames[category][lang];
            } else if (uiTexts.presetCustomCategories[category]) {
                displayText = uiTexts.presetCustomCategories[category][lang] || uiTexts.presetCustomCategories[category]['zh'];
            }
            btn.textContent = displayText;

            if (category === activeCategory) {
                btn.classList.add('active');
            }
            btn.addEventListener('click', () => clickHandler(category));
            wrapper.appendChild(btn);
        });

        if (isPrimary) {
            const allBtn = rowElement.querySelector('#fixed-all-btn');
            allBtn.classList.toggle('active', activeCategory === 'All');
        }
    }
    function renderPrimaryCategories() { renderCategoryRow(primaryCategoryRow, primaryCategories, primaryCategoryPage, activePrimaryCategory, handlePrimaryCategorySelect, true); }
    function renderSecondaryCategories() { renderCategoryRow(secondaryCategoryRow, Array.from(secondaryCategories), secondaryCategoryPage, activeSecondaryCategory, handleSecondaryCategorySelect, false); }
    async function selectAndLoadWildcard(categoryName) {
        activeSecondaryCategory = categoryName;
        renderSecondaryCategories();
        await loadAndDisplayWildcardContent(categoryName);
    }

    // [新版本] handlePrimaryCategorySelect 函数
    async function handlePrimaryCategorySelect(category) {
        console.log(`一级分类选择: ${category}`);
        activePrimaryCategory = category;
        activeSecondaryCategory = null; // 重置二级分类选择
        secondaryCategories.clear();
        secondaryCategoryPage = 1;

        // [核心修改] 逻辑重构，不再使用 "内置分类-" 前缀
        if (category === 'Wildcard') {
            // --- 通配符逻辑 (保持不变) ---
            const wildcardList = Object.keys(wildcardFilenames).sort();
            secondaryCategories = new Set(wildcardList);
            
            if (wildcardList.length > 0) {
                const firstWildcard = wildcardList[0];
                console.log(`通配符分类被选中，自动加载第一个文件: ${firstWildcard}`);
                await selectAndLoadWildcard(firstWildcard);
            }
        } 
        else if (uiTexts.presetCustomCategories[category]) {
            // --- 新的自定义分类逻辑 ---
            const secondarySet = new Set();
            // 1. 筛选出所有属于当前一级分类的标签
            const relevantTags = allTags.filter(tag => tag.customCategory === category);
            
            // 2. 从这些标签中提取二级分类
            relevantTags.forEach(tag => {
                // 如果二级分类字段为空，则归入"未分类"；否则使用其自身的值
                secondarySet.add(tag.secondaryCategory || '未分类');
            });

            if (secondarySet.size > 0) {
                secondaryCategories = new Set(Array.from(secondarySet).sort());
            }
        }
        
        // 更新一级分类的分页（逻辑不变）
        const categoryIndex = primaryCategories.indexOf(category);
        if (categoryIndex !== -1) {
            primaryCategoryPage = Math.ceil((categoryIndex + 1) / CATEGORIES_PER_PAGE);
        }
        if(category === 'All') primaryCategoryPage = 1;

        // 重新渲染分类按钮
        renderPrimaryCategories();
        renderSecondaryCategories();
        
        // 如果不是通配符分类，则立即应用筛选并显示结果
        if (category !== 'Wildcard') {
            applyFiltersAndRender();
        }
    }

    // [修改] 替换为这个新的 handleSecondaryCategorySelect 函数
    async function handleSecondaryCategorySelect(category) {
        // 确定新的活动分类。如果点击的是当前已激活的，则取消选择 (newActiveCategory 为 null)
        const newActiveCategory = activeSecondaryCategory === category ? null : category;
        
        // 更新活动二级分类的状态
        activeSecondaryCategory = newActiveCategory;

        // 首先，无条件地重新渲染二级分类按钮，以正确反映高亮状态
        renderSecondaryCategories();

        // 如果没有新的活动分类（即用户取消了选择），则恢复默认过滤并返回
        if (!newActiveCategory) {
            applyFiltersAndRender();
            return;
        }

        // [核心逻辑] 根据当前的一级分类，决定下一步操作
        if (activePrimaryCategory === 'Wildcard') {
            // --- 这是通配符分类的专属逻辑 ---
            
            // 1. 加载并显示该通配符文件的内容
            await loadAndDisplayWildcardContent(newActiveCategory);

            // 2. [重要] 为通配符创建占位符并添加到已选区
            const wildcardTag = {
                name: `__${newActiveCategory}__`,
                translation: `__${wildcardFilenames[newActiveCategory] || newActiveCategory}__`,
                category: -1, 
                isWildcardPlaceholder: true
            };
            toggleTagSelection(wildcardTag);

        } else {
            // --- 这是所有其他（非通配符）二级分类的逻辑 ---
            
            // 只需要根据新选择的二级分类来过滤标签列表即可
            // 不执行任何文件加载，也不添加任何东西到已选区
            applyFiltersAndRender();
        }
    }

    async function loadAndDisplayWildcardContent(filename) {
        try {
            const response = await fetch(`${webpath}/wildcards/${filename}.txt`);
            if (!response.ok) throw new Error('File not found');
            const text = await response.text();
            const lines = text.split('\n').filter(line => line.trim() !== '');

            filteredTags = lines.map(line => {
                const name = line.trim();
                return { name: name, translation: wildcardWordTranslations[name] || '', category: -2, count: 0, isWildcard: true };
            });
            currentPage = 1;
            renderTags();
            renderPagination();
        } catch (error) {
            console.error(`加载通配符文件 ${filename}.txt 失败:`, error);
            filteredTags = [];
            renderTags();
            renderPagination();
        }
    }

    // [新版本] applyFiltersAndRender 函数
    function applyFiltersAndRender() {
        // 如果正在显示通配符文件内容，则不执行常规过滤
        if (activePrimaryCategory === 'Wildcard' && activeSecondaryCategory) {
            return;
        }
        
        const query = searchInput.value.toLowerCase().trim();
        let tempFilteredTags = [...allTags, ...searchableWildcardTags];

        // [核心修改 1] 扩大搜索范围，包含 "remarks" 字段
        if (query) {
            tempFilteredTags = tempFilteredTags.filter(tag => 
                tag.name.toLowerCase().includes(query) || 
                (tag.aliases && tag.aliases.toLowerCase().includes(query)) || 
                (tag.translation && tag.translation.toLowerCase().includes(query)) || 
                (tag.customCategory && tag.customCategory.toLowerCase().includes(query)) ||
                (tag.remarks && tag.remarks.toLowerCase().includes(query)) // <-- 已包含备注搜索
            );
        }

        // [核心修改 2] 简化分类过滤逻辑
        if (activePrimaryCategory && activePrimaryCategory !== 'All') {
            switch (activePrimaryCategory) {
                // --- 标准分类 (逻辑不变) ---
                case 'General':   tempFilteredTags = tempFilteredTags.filter(t => t.category === 0); break;
                case 'Character': tempFilteredTags = tempFilteredTags.filter(t => t.category === 4); break;
                case 'Copyright': tempFilteredTags = tempFilteredTags.filter(t => t.category === 3); break;
                case 'Artist':    tempFilteredTags = tempFilteredTags.filter(t => t.category === 1); break;
                case 'Meta':      tempFilteredTags = tempFilteredTags.filter(t => t.category === 5); break;
                case 'Kontext':   tempFilteredTags = tempFilteredTags.filter(t => t.category === 6); break;
                case 'Custom':    tempFilteredTags = tempFilteredTags.filter(t => t.category === 9); break;
                case 'Wildcard':  // (逻辑不变)
                    if (!query) { tempFilteredTags = []; } 
                    else { tempFilteredTags = tempFilteredTags.filter(t => t.isWildcard); }
                    break;
                // --- 新的自定义分类过滤逻辑 ---
                default:
                    if (uiTexts.presetCustomCategories[activePrimaryCategory]) {
                        // 1. 按一级分类筛选
                        tempFilteredTags = tempFilteredTags.filter(t => t.customCategory === activePrimaryCategory);
                        
                        // 2. 如果有二级分类被激活，则进一步筛选
                        if (activeSecondaryCategory) {
                            if (activeSecondaryCategory === '未分类') {
                                // 如果选择的是"未分类"，则筛选出 secondaryCategory 为空的标签
                                tempFilteredTags = tempFilteredTags.filter(t => !t.secondaryCategory);
                            } else {
                                // 否则，精确匹配二级分类名称
                                tempFilteredTags = tempFilteredTags.filter(t => t.secondaryCategory === activeSecondaryCategory);
                            }
                        }
                    }
                    break;
            }
        }

        // NSFW 过滤 (逻辑不变)
        if (isNsfwFilterActive) {
            tempFilteredTags = tempFilteredTags.filter(tag => !tag.customCategory.toLowerCase().includes('内置分类-禁'));
        }

        // 去重、排序和渲染 (逻辑不变)
        const uniqueNames = new Set();
        const uniqueTags = tempFilteredTags.filter(tag => {
            if (uniqueNames.has(tag.name)) return false;
            else { uniqueNames.add(tag.name); return true; }
        });
        uniqueTags.sort((a, b) => (b.count || 0) - (a.count || 0));
        filteredTags = uniqueTags;
        currentPage = 1; 
        renderTags(); 
        renderPagination(); 
    }

    function renderTags() {
        tagDisplayContainer.innerHTML = ''; 
        const startIndex = (currentPage - 1) * TAGS_PER_PAGE;
        const tagsToRender = filteredTags.slice(startIndex, startIndex + TAGS_PER_PAGE); 
        tagsToRender.forEach(tag => {
            const tagEl = document.createElement('div');
            const lang = displayEnglishOnly ? 'en' : 'zh';
            let displayText = tag.name.replace(/_/g, ' ');
            if (lang === 'zh' && tag.translation) displayText = tag.translation;
            
            tagEl.className = 'tag-item tag-interactive px-2 py-1 rounded-md cursor-pointer text-xs flex items-center justify-center';
            tagEl.dataset.category = tag.category; 
            tagEl.dataset.tagName = tag.name; 
            tagEl.title = generateTagTitle(tag);             

            if (tag.isCustom) tagEl.classList.add('custom-tag');
            else if (tag.isWildcard) tagEl.classList.add('wildcard-tag');
            
            const textSpan = document.createElement('span');
            textSpan.textContent = displayText;
            textSpan.style.cssText = 'overflow: hidden; white-space: nowrap; text-overflow: ellipsis; min-width: 0;';
            tagEl.appendChild(textSpan);
            if (selectedTags.some(st => st.name === tag.name)) tagEl.classList.add('selected');
            tagEl.addEventListener('click', () => toggleTagSelection(tag)); 
            tagDisplayContainer.appendChild(tagEl);
        });
    }
    function renderSelectedTags() {
        selectedTagsContainer.innerHTML = ''; 
        selectedTags.forEach(tag => {
            const tagEl = document.createElement('div');
            const lang = displayEnglishOnly ? 'en' : 'zh';
            let displayText = tag.name.replace(/_/g, ' ');
            if (lang === 'zh' && tag.translation) displayText = tag.translation;

            tagEl.className = 'tag-item tag-interactive px-2 py-1 rounded-md cursor-pointer text-xs flex items-center justify-center min-w-[50px] max-w-[250px]';
            tagEl.dataset.category = tag.category; 
            tagEl.title = generateTagTitle(tag);
            
            // [修改点] 检查并应用特殊样式
            if (tag.isUnmatched) {
                tagEl.classList.add('unmatched-tag');
            } else if (tag.isCustom) {
                tagEl.classList.add('custom-tag');
            } else if (tag.isWildcardPlaceholder || tag.isWildcard) {
                tagEl.classList.add('wildcard-tag');
            }

            if (tag.isCustom) tagEl.classList.add('custom-tag');
            else if (tag.isWildcardPlaceholder || tag.isWildcard) tagEl.classList.add('wildcard-tag');

            const textSpan = document.createElement('span');
            textSpan.textContent = displayText;
            textSpan.style.cssText = 'overflow: hidden; white-space: nowrap; text-overflow: ellipsis; min-width: 0;';
            tagEl.appendChild(textSpan);
            tagEl.addEventListener('click', () => toggleTagSelection(tag)); 
            selectedTagsContainer.appendChild(tagEl);
        });
    }
    function renderPagination() {
        paginationContainer.innerHTML = '';
        const totalPages = Math.ceil(filteredTags.length / TAGS_PER_PAGE);
        if (totalPages <= 1) return;
        const createButton = (html, disabled, onClick) => {
            const btn = document.createElement('button');
            btn.innerHTML = html;
            btn.className = 'btn p-2 rounded-lg h-8 w-8 flex items-center justify-center';
            btn.disabled = disabled;
            if (!disabled) btn.onclick = onClick;
            return btn;
        };
        paginationContainer.appendChild(createButton('<i class="fa-solid fa-chevron-left"></i>', currentPage === 1, () => { currentPage--; renderTags(); renderPagination(); }));
        const pageInfo = document.createElement('span');
        pageInfo.textContent = `${currentPage} / ${totalPages}`;
        pageInfo.className = 'text-sm';
        paginationContainer.appendChild(pageInfo);
        paginationContainer.appendChild(createButton('<i class="fa-solid fa-chevron-right"></i>', currentPage === totalPages, () => { currentPage++; renderTags(); renderPagination(); }));
    }
    function toggleTagSelection(tag) {
        const index = selectedTags.findIndex(st => st.name === tag.name);
        if (index > -1) { selectedTags.splice(index, 1); } 
        else { selectedTags.push(tag); }
        renderSelectedTags(); 
        const displayedTagElement = tagDisplayContainer.querySelector(`[data-tag-name="${tag.name}"]`);
        if (displayedTagElement) { displayedTagElement.classList.toggle('selected', index === -1); }
    }

    function generateTagTitle(tag) {
        // [修改点] 在函数最开头添加判断
        if (tag.isUnmatched) {
            return `未匹配的标签: ${tag.name}\n该标签将按原样保留和导出。`;
        }
        if (tag.isWildcardPlaceholder) return `通配符: ${tag.name}`;
        if (tag.isWildcard) return `通配符词条\n英文: ${tag.name}\n中文: ${tag.translation || '无'}`;

        const lang = displayEnglishOnly ? 'en' : 'zh';
        const getCategoryName = (categoryCode) => {
            const map = {0: 'General', 1: 'Artist', 3: 'Copyright', 4: 'Character', 5: 'Meta', 6: 'Kontext', 9: 'Custom'};
            const key = map[categoryCode];
            return uiTexts.primaryCategoryNames[key]?.[lang] || `${uiTexts.tagTitleDefaults[lang].unknownCategory} (${categoryCode})`;
        };

        return [
            `英文: ${tag.name}`, `中文: ${tag.translation || uiTexts.tagTitleDefaults[lang].noTranslation}`,
            `别名: ${tag.aliases || uiTexts.tagTitleDefaults[lang].noAliases}`,
            `${lang === 'zh' ? '类别' : 'Category'}: ${getCategoryName(tag.category)}`,
            `${lang === 'zh' ? '帖子数量' : 'Post Count'}: ${tag.count.toLocaleString()}`,
            `${lang === 'zh' ? '自定义分类' : 'Custom Category'}: ${tag.customCategory || uiTexts.tagTitleDefaults[lang].noCustomCategory}`,
            `${lang === 'zh' ? '二级分类' : 'Secondary'}: ${tag.secondaryCategory || uiTexts.tagTitleDefaults[lang].noSecondaryCategory}`,
            `${lang === 'zh' ? '备注' : 'Remarks'}: ${tag.remarks || uiTexts.tagTitleDefaults[lang].noTranslation}` // <-- 新增这一行，'无'/'None'的翻译可以复用
        ].join('\n');
    }

    function formatTags() {
        return selectedTags.map(tag => {
            let tagName = tag.name;

            // [修改点] 在函数开头添加判断
            if (tag.isUnmatched) {
                return tagName; // 对于未匹配标签，直接原样返回
            }
            if (tag.isWildcardPlaceholder || tag.isWildcard) return tagName;
            tagName = tagName.replace(/\(/g, '\\(').replace(/\)/g, '\\)');
            tagName = tagName.replace(/_/g, ' '); 
            if (activeFormat === 'full') tagName = `(${tagName}:${FIXED_WEIGHT})`; 
            return tagName;
        }).join(', '); 
    }

    function copyTagsToClipboard() {
        const formattedString = formatTags(); 
        if (!formattedString) return; 
        const targetId = activeTarget === 'positive' ? 'positive_prompt' : 'negative_prompt';
        const promptTextarea = document.querySelector(`#${targetId} textarea`);

        if (promptTextarea) {
            if (activeAction === 'append') {
                promptTextarea.value += (promptTextarea.value ? ', ' : '') + formattedString;
            } else { promptTextarea.value = formattedString; }
            promptTextarea.dispatchEvent(new Event('input', { bubbles: true }));
        } else { console.warn(`未能找到ID为 "${targetId}" 的Gradio文本框。`); }
        // navigator.clipboard.writeText(formattedString).catch(err => console.error('复制失败', err));
    }

    function updateUIText(lang) {
        searchInput.placeholder = uiTexts.searchInputPlaceholder[lang];
        draggableHandle.textContent = uiTexts.draggableHandleText[lang];
        copyBtn.innerHTML = `<i class="fa-solid fa-copy"></i> ${uiTexts.buttonTitles.copy[lang]}`;
        const allBtn = primaryCategoryRow.querySelector('#fixed-all-btn');
        if (allBtn) allBtn.textContent = uiTexts.primaryCategoryNames['All'][lang];

        const updateButtonGroupText = (groupElement, labels) => {
            groupElement.querySelectorAll('button').forEach(btn => {
                const value = btn.dataset.value;
                if (labels[value]) btn.textContent = labels[value][lang];
            });
        };
        updateButtonGroupText(formatBtnGroup, uiTexts.formatButtonLabels);
        updateButtonGroupText(actionBtnGroup, uiTexts.actionButtonLabels);
        updateButtonGroupText(targetBtnGroup, uiTexts.targetButtonLabels);

        resetSearchBtn.title = uiTexts.buttonTitles.resetSearch[lang];
        nsfwFilterBtn.title = uiTexts.buttonTitles.nsfwFilter[lang];
        clearAllBtn.title = uiTexts.buttonTitles.clearAll[lang];
        toggleLanguageBtn.title = uiTexts.buttonTitles.toggleLanguage[lang];

        // 只有在数据加载后才执行渲染，否则会出错
        if (isDataLoaded) {
            renderPrimaryCategories();
            renderSecondaryCategories();
            renderTags();
            renderSelectedTags();
        }
    }
    function applyThemeFromUrl() {
        const theme = new URLSearchParams(window.location.search).get('__theme');
        document.documentElement.setAttribute('data-theme', theme === 'dark' ? 'dark' : 'light');
    }

    // --- 启动应用 ---
    init(); // 1. 初始化UI骨架
    applyThemeFromUrl(); 
    setupEventListeners(); // 2. 绑定基础事件
    positionDraggableContainer();
    if (isNsfwFilterActive) nsfwFilterBtn.classList.add('active');
    
    // [优化 5] 设置MutationObserver来监听面板显示，以触发懒加载
    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            // 我们只关心 style 属性的变化
            if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                // 当面板从 'none' 变为可见状态时 (外部脚本会设置 display: 'block')
                // 并且数据尚未加载
                if (draggableContainer.style.display !== 'none' && !isDataLoaded) {
                    console.log("检测到面板变为可见，首次触发数据加载。");
                    triggerDataLoadAndDisplay(); // 调用我们的懒加载函数
                    
                    // 可选：数据只需要加载一次，之后可以断开观察者以节省资源
                    // observer.disconnect(); // 但保持连接也无害，因为 isDataLoaded 标志会阻止重复加载
                }
            }
        }
    });

    // 开始观察 draggableContainer 的属性变化
    observer.observe(draggableContainer, { attributes: true });


    new Sortable(selectedTagsContainer, {
        animation: 150, 
        ghostClass: 'ghost-class', 
        onEnd: (evt) => {
            const movedTag = selectedTags.splice(evt.oldIndex, 1)[0];
            selectedTags.splice(evt.newIndex, 0, movedTag);
        }
    });

    // [优化 6] 移除原有的立即执行的异步加载块
    /*
    (async () => {
        await loadAllData();
        renderPrimaryCategories();
        applyFiltersAndRender();
        updateUIText(displayEnglishOnly ? 'en' : 'zh');
    })();
    */
   // 现在，所有数据加载和后续渲染都由 MutationObserver 触发。
}

// ==================== 移动设备检测 ====================
function isMobileDevice() {
    return /android|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(navigator.userAgent.toLowerCase());
}

// --- 轮询机制 (无变化) ---
window.addEventListener('load', () => {
    if (isMobileDevice()) {
        console.log("移动设备，不加载标签助手。");
        return;
    }
    console.log("Window loaded. Initializing tag assistant logic and starting to poll for Gradio app...");

    try {
        initializeTagAssistantLogic(); // 立即执行，创建UI骨架和监听器

        let attempts = 0;
        const maxAttempts = 50;
        const intervalId = setInterval(() => {
            const gradioContainer = typeof gradioApp === 'function' ? gradioApp() : null;
            
            if (gradioContainer) {
                console.log("Gradio app found! Appending tag assistant.");
                gradioContainer.appendChild(appRootInstance);
                clearInterval(intervalId);
            } else {
                attempts++;
                if (attempts >= maxAttempts) {
                    console.error("Gradio app not found after multiple attempts. Appending to body as a fallback.");
                    document.body.appendChild(appRootInstance);
                    clearInterval(intervalId);
                }
            }
        }, 100);

    } catch (e) {
        console.error("Failed to initialize Tag Assistant Logic:", e);
    }
});
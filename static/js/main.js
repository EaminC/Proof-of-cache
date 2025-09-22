// 主应用程序

class TransformerVisualizerApp {
    constructor() {
        this.initialized = false;
        this.visualizers = {};
        this.currentAttentionLayer = 0;
        this.currentAttentionHead = 0;
        
        this.init();
    }

    /**
     * 初始化应用
     */
    init() {
        // 等待DOM加载完成
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initializeApp());
        } else {
            this.initializeApp();
        }
    }

    /**
     * 初始化应用组件
     */
    initializeApp() {
        try {
            // 初始化可视化器
            this.initializeVisualizers();
            
            // 绑定事件监听器
            this.bindEventListeners();
            
            // 设置状态管理器事件
            this.setupStateListeners();
            
            // 恢复之前的设置
            this.restoreSettings();
            
            this.initialized = true;
            console.log('🎉 Transformer可视化器已初始化完成');
            
        } catch (error) {
            console.error('初始化失败:', error);
            window.errorHandler.showError(error, '应用初始化');
        }
    }

    /**
     * 初始化可视化器
     */
    initializeVisualizers() {
        // Token序列可视化器
        const tokensContainer = document.getElementById('tokens-container');
        this.visualizers.tokens = new TokenVisualizer(tokensContainer);
        window.tokenVisualizer = this.visualizers.tokens; // 全局引用

        // 模型结构可视化器
        const modelStructureContainer = document.getElementById('model-structure');
        this.visualizers.modelStructure = new ModelStructureVisualizer(modelStructureContainer);

        // 注意力热图可视化器
        const attentionContainer = document.getElementById('attention-heatmap');
        this.visualizers.attention = new AttentionHeatmapVisualizer(attentionContainer);

        // Token详情可视化器
        const tokenDetailsContainer = document.getElementById('token-details');
        this.visualizers.tokenDetails = new TokenDetailVisualizer(tokenDetailsContainer);

        // 状态轨迹可视化器
        const statesContainer = document.getElementById('states-chart');
        this.visualizers.stateTrajectory = new StateTrajectoryVisualizer(statesContainer);
    }

    /**
     * 绑定事件监听器
     */
    bindEventListeners() {
        // 加载模型按钮
        const loadModelBtn = document.getElementById('load-model-btn');
        loadModelBtn.addEventListener('click', () => this.loadModel());

        // 分析按钮
        const analyzeBtn = document.getElementById('analyze-btn');
        analyzeBtn.addEventListener('click', () => this.analyzeText());

        // 模型选择下拉框
        const modelSelect = document.getElementById('model-select');
        modelSelect.addEventListener('change', () => this.onModelSelectionChange());

        // 注意力层级和头部选择
        const attentionLayerSelect = document.getElementById('attention-layer-select');
        const attentionHeadSelect = document.getElementById('attention-head-select');
        
        attentionLayerSelect.addEventListener('change', () => this.onAttentionLayerChange());
        attentionHeadSelect.addEventListener('change', () => this.onAttentionHeadChange());

        // 目标tokens复选框
        const analyzeAllTokens = document.getElementById('analyze-all-tokens');
        analyzeAllTokens.addEventListener('change', () => this.onAnalyzeAllTokensChange());

        // 侧边面板关闭按钮
        const closePanelBtn = document.getElementById('close-panel');
        closePanelBtn.addEventListener('click', () => this.closeSidePanel());

        // 键盘快捷键
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));

        // 窗口大小改变
        window.addEventListener('resize', Utils.debounce(() => this.onWindowResize(), 300));
    }

    /**
     * 设置状态管理器事件监听
     */
    setupStateListeners() {
        const state = window.modelState;

        // 模型加载完成
        state.on('modelChanged', (modelName, modelInfo) => {
            this.onModelLoaded(modelName, modelInfo);
        });

        // 分析完成
        state.on('analysisComplete', (result) => {
            this.onAnalysisComplete(result);
        });

        // Token选择
        state.on('tokenSelected', (tokenPosition) => {
            this.onTokenSelected(tokenPosition);
        });

        // Token详情加载完成
        state.on('tokenDetailsLoaded', (tokenPosition, details) => {
            this.onTokenDetailsLoaded(tokenPosition, details);
        });

        // 层级选择
        state.on('layerSelected', (layerIndex) => {
            this.onLayerSelected(layerIndex);
        });

        // 错误处理
        state.on('error', (errorInfo) => {
            this.onError(errorInfo);
        });
    }

    /**
     * 加载模型
     */
    async loadModel() {
        try {
            const modelSelect = document.getElementById('model-select');
            const modelStatus = document.getElementById('model-status');
            const loadModelBtn = document.getElementById('load-model-btn');
            
            const modelName = modelSelect.value;
            
            // 更新UI状态
            loadModelBtn.disabled = true;
            Utils.showStatus(modelStatus, '正在加载模型...', 'loading');
            Utils.showLoading(`正在加载 ${modelName} 模型...`);

            // 调用API加载模型
            const response = await window.api.loadModel(modelName);
            
            if (response.success) {
                // 更新状态管理器
                window.modelState.setModel(modelName, response.model_info);
                
                // 更新UI
                Utils.showStatus(modelStatus, `模型 ${modelName} 加载成功`, 'success');
                
                // 保存设置
                Utils.Storage.set('selectedModel', modelName);
                
            } else {
                throw new Error(response.error || '模型加载失败');
            }

        } catch (error) {
            const modelStatus = document.getElementById('model-status');
            Utils.showStatus(modelStatus, `加载失败: ${error.message}`, 'error');
            window.errorHandler.showError(error, '模型加载');
            
        } finally {
            const loadModelBtn = document.getElementById('load-model-btn');
            loadModelBtn.disabled = false;
            Utils.hideLoading();
        }
    }

    /**
     * 分析文本
     */
    async analyzeText() {
        try {
            if (!window.modelState.hasModel()) {
                throw new Error('请先加载模型');
            }

            const inputText = document.getElementById('input-text').value.trim();
            const analyzeAllTokens = document.getElementById('analyze-all-tokens').checked;
            const targetTokensInput = document.getElementById('target-tokens').value.trim();
            const analyzeBtn = document.getElementById('analyze-btn');

            if (!inputText) {
                throw new Error('请输入要分析的文本');
            }

            // 准备目标tokens
            let targetTokens = null;
            if (!analyzeAllTokens && targetTokensInput) {
                targetTokens = targetTokensInput.split(',').map(t => t.trim()).filter(t => t);
            }

            // 更新UI状态
            analyzeBtn.disabled = true;
            Utils.showLoading('正在分析文本...');

            // 调用API分析文本
            const response = await window.api.analyzeText(inputText, targetTokens);
            
            if (response.success) {
                // 更新状态管理器
                window.modelState.setAnalysisResult(response);
                
                // 保存输入历史
                this.saveInputHistory(inputText);
                
            } else {
                throw new Error(response.error || '文本分析失败');
            }

        } catch (error) {
            window.errorHandler.showError(error, '文本分析');
            
        } finally {
            const analyzeBtn = document.getElementById('analyze-btn');
            analyzeBtn.disabled = false;
            Utils.hideLoading();
        }
    }

    /**
     * 模型加载完成回调
     * @param {string} modelName 模型名称
     * @param {Object} modelInfo 模型信息
     */
    async onModelLoaded(modelName, modelInfo) {
        try {
            // 获取模型结构
            const structureResponse = await window.api.getModelStructure();
            if (structureResponse.success) {
                this.visualizers.modelStructure.renderStructure(structureResponse.structure);
                
                // 更新注意力层级选择器
                this.updateAttentionLayerSelector(structureResponse.structure.num_layers);
            }

            // 更新注意力头部选择器
            this.updateAttentionHeadSelector(modelInfo.num_attention_heads || 12);

        } catch (error) {
            console.error('获取模型结构失败:', error);
        }
    }

    /**
     * 分析完成回调
     * @param {Object} result 分析结果
     */
    async onAnalysisComplete(result) {
        try {
            // 渲染tokens
            this.visualizers.tokens.renderTokens(result.tokens);
            
            // 更新token计数显示
            const tokenCount = document.getElementById('token-count');
            tokenCount.textContent = `总tokens: ${result.tokens.length}`;

            // 获取注意力矩阵
            const attentionResponse = await window.api.getAttentionMatrix();
            if (attentionResponse.success) {
                this.visualizers.attention.setData(attentionResponse.attention_matrices, result.tokens);
            }

        } catch (error) {
            console.error('处理分析结果失败:', error);
        }
    }

    /**
     * Token选择回调
     * @param {number} tokenPosition token位置
     */
    async onTokenSelected(tokenPosition) {
        try {
            Utils.showLoading('加载Token详情...');

            // 检查缓存
            let details = window.modelState.getTokenDetails(tokenPosition);
            
            if (!details) {
                // 从API获取详情
                const response = await window.api.getTokenDetails(tokenPosition);
                if (response.success) {
                    details = response;
                    window.modelState.setTokenDetails(tokenPosition, details);
                } else {
                    throw new Error(response.error || '获取Token详情失败');
                }
            }

            // 显示详情
            this.visualizers.tokenDetails.showDetails(details);
            this.visualizers.stateTrajectory.showTrajectory(details);

        } catch (error) {
            window.errorHandler.showError(error, 'Token详情获取');
            
        } finally {
            Utils.hideLoading();
        }
    }

    /**
     * Token详情加载完成回调
     * @param {number} tokenPosition token位置
     * @param {Object} details 详情数据
     */
    onTokenDetailsLoaded(tokenPosition, details) {
        // 高亮相关层级
        if (details.layer_data) {
            const activeLayers = Object.keys(details.layer_data).map(Number);
            this.visualizers.modelStructure.highlightLayers(activeLayers);
        }
    }

    /**
     * 层级选择回调
     * @param {number} layerIndex 层级索引
     */
    onLayerSelected(layerIndex) {
        // 更新注意力层级选择器
        const attentionLayerSelect = document.getElementById('attention-layer-select');
        attentionLayerSelect.value = layerIndex;
        this.currentAttentionLayer = layerIndex;
        
        // 更新注意力热图
        this.visualizers.attention.setLayerAndHead(this.currentAttentionLayer, this.currentAttentionHead);
    }

    /**
     * 错误处理回调
     * @param {Object} errorInfo 错误信息
     */
    onError(errorInfo) {
        console.error('应用错误:', errorInfo);
    }

    /**
     * 注意力层级改变
     */
    onAttentionLayerChange() {
        const attentionLayerSelect = document.getElementById('attention-layer-select');
        this.currentAttentionLayer = parseInt(attentionLayerSelect.value);
        this.visualizers.attention.setLayerAndHead(this.currentAttentionLayer, this.currentAttentionHead);
    }

    /**
     * 注意力头部改变
     */
    onAttentionHeadChange() {
        const attentionHeadSelect = document.getElementById('attention-head-select');
        this.currentAttentionHead = parseInt(attentionHeadSelect.value);
        this.visualizers.attention.setLayerAndHead(this.currentAttentionLayer, this.currentAttentionHead);
    }

    /**
     * 分析所有tokens选项改变
     */
    onAnalyzeAllTokensChange() {
        const analyzeAllTokens = document.getElementById('analyze-all-tokens');
        const targetTokensInput = document.getElementById('target-tokens');
        
        targetTokensInput.disabled = analyzeAllTokens.checked;
        if (analyzeAllTokens.checked) {
            targetTokensInput.placeholder = '将分析所有tokens';
        } else {
            targetTokensInput.placeholder = '目标tokens（用逗号分隔）';
        }
    }

    /**
     * 模型选择改变
     */
    onModelSelectionChange() {
        // 清除之前的状态
        window.modelState.setModel(null, null);
        this.clearAllVisualizations();
    }

    /**
     * 关闭侧边面板
     */
    closeSidePanel() {
        const sidePanel = document.getElementById('side-panel');
        sidePanel.classList.remove('open');
    }

    /**
     * 处理键盘快捷键
     * @param {KeyboardEvent} e 键盘事件
     */
    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + Enter: 分析文本
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            this.analyzeText();
        }
        
        // Escape: 关闭侧边面板
        if (e.key === 'Escape') {
            this.closeSidePanel();
        }
    }

    /**
     * 窗口大小改变处理
     */
    onWindowResize() {
        // 重新渲染图表
        Object.values(this.visualizers).forEach(visualizer => {
            if (typeof visualizer.onResize === 'function') {
                visualizer.onResize();
            }
        });
    }

    /**
     * 更新注意力层级选择器
     * @param {number} numLayers 层数
     */
    updateAttentionLayerSelector(numLayers) {
        const select = document.getElementById('attention-layer-select');
        select.innerHTML = '';
        
        for (let i = 0; i < numLayers; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `Layer ${i}`;
            select.appendChild(option);
        }
    }

    /**
     * 更新注意力头部选择器
     * @param {number} numHeads 头数
     */
    updateAttentionHeadSelector(numHeads) {
        const select = document.getElementById('attention-head-select');
        select.innerHTML = '';
        
        for (let i = 0; i < numHeads; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `Head ${i}`;
            select.appendChild(option);
        }
    }

    /**
     * 清除所有可视化
     */
    clearAllVisualizations() {
        Object.values(this.visualizers).forEach(visualizer => {
            if (typeof visualizer.clear === 'function') {
                visualizer.clear();
            }
        });
        
        // 重置UI状态
        const tokenCount = document.getElementById('token-count');
        const selectedToken = document.getElementById('selected-token');
        
        tokenCount.textContent = '总tokens: 0';
        selectedToken.textContent = '选中token: 无';
    }

    /**
     * 保存输入历史
     * @param {string} text 输入文本
     */
    saveInputHistory(text) {
        let history = Utils.Storage.get('inputHistory', []);
        
        // 避免重复
        history = history.filter(item => item !== text);
        history.unshift(text);
        
        // 保持最多10条历史
        if (history.length > 10) {
            history = history.slice(0, 10);
        }
        
        Utils.Storage.set('inputHistory', history);
    }

    /**
     * 恢复之前的设置
     */
    restoreSettings() {
        // 恢复模型选择
        const selectedModel = Utils.Storage.get('selectedModel');
        if (selectedModel) {
            const modelSelect = document.getElementById('model-select');
            if (modelSelect.querySelector(`option[value="${selectedModel}"]`)) {
                modelSelect.value = selectedModel;
            }
        }

        // 恢复输入历史（可选实现）
        // this.restoreInputHistory();
    }
}

// 启动应用
window.app = new TransformerVisualizerApp();
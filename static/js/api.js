// API接口模块

class API {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }

    /**
     * 通用请求方法
     * @param {string} endpoint API端点
     * @param {Object} options 请求选项
     * @returns {Promise}
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    /**
     * GET请求
     * @param {string} endpoint API端点
     * @param {Object} params 查询参数
     * @returns {Promise}
     */
    async get(endpoint, params = {}) {
        const searchParams = new URLSearchParams(params);
        const url = searchParams.toString() ? `${endpoint}?${searchParams}` : endpoint;
        
        return this.request(url, {
            method: 'GET'
        });
    }

    /**
     * POST请求
     * @param {string} endpoint API端点
     * @param {Object} data 请求数据
     * @returns {Promise}
     */
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * 加载模型
     * @param {string} modelName 模型名称
     * @returns {Promise}
     */
    async loadModel(modelName) {
        return this.post('/load_model', { model_name: modelName });
    }

    /**
     * 分析文本
     * @param {string} text 输入文本
     * @param {Array|null} targetTokens 目标tokens
     * @returns {Promise}
     */
    async analyzeText(text, targetTokens = null) {
        return this.post('/analyze_text', {
            text: text,
            target_tokens: targetTokens
        });
    }

    /**
     * 获取token详情
     * @param {number} tokenPosition token位置
     * @returns {Promise}
     */
    async getTokenDetails(tokenPosition) {
        return this.get(`/get_token_details/${tokenPosition}`);
    }

    /**
     * 获取注意力矩阵
     * @returns {Promise}
     */
    async getAttentionMatrix() {
        return this.get('/get_attention_matrix');
    }

    /**
     * 获取模型结构
     * @returns {Promise}
     */
    async getModelStructure() {
        return this.get('/get_model_structure');
    }
}

// 创建全局API实例
window.api = new API();

// 模型状态管理
class ModelState {
    constructor() {
        this.currentModel = null;
        this.modelInfo = null;
        this.analysisResult = null;
        this.selectedToken = null;
        this.tokenDetails = new Map(); // 缓存token详情
        this.eventEmitter = new Utils.EventEmitter();
    }

    /**
     * 设置当前模型
     * @param {string} modelName 模型名称
     * @param {Object} modelInfo 模型信息
     */
    setModel(modelName, modelInfo) {
        this.currentModel = modelName;
        this.modelInfo = modelInfo;
        this.analysisResult = null;
        this.selectedToken = null;
        this.tokenDetails.clear();
        
        this.eventEmitter.emit('modelChanged', modelName, modelInfo);
    }

    /**
     * 设置分析结果
     * @param {Object} result 分析结果
     */
    setAnalysisResult(result) {
        this.analysisResult = result;
        this.selectedToken = null;
        this.tokenDetails.clear();
        
        this.eventEmitter.emit('analysisComplete', result);
    }

    /**
     * 选择token
     * @param {number} tokenPosition token位置
     */
    selectToken(tokenPosition) {
        if (this.selectedToken === tokenPosition) return;
        
        this.selectedToken = tokenPosition;
        this.eventEmitter.emit('tokenSelected', tokenPosition);
    }

    /**
     * 缓存token详情
     * @param {number} tokenPosition token位置
     * @param {Object} details 详情数据
     */
    setTokenDetails(tokenPosition, details) {
        this.tokenDetails.set(tokenPosition, details);
        this.eventEmitter.emit('tokenDetailsLoaded', tokenPosition, details);
    }

    /**
     * 获取缓存的token详情
     * @param {number} tokenPosition token位置
     * @returns {Object|null}
     */
    getTokenDetails(tokenPosition) {
        return this.tokenDetails.get(tokenPosition) || null;
    }

    /**
     * 监听事件
     * @param {string} event 事件名
     * @param {Function} callback 回调函数
     */
    on(event, callback) {
        this.eventEmitter.on(event, callback);
    }

    /**
     * 移除事件监听
     * @param {string} event 事件名
     * @param {Function} callback 回调函数
     */
    off(event, callback) {
        this.eventEmitter.off(event, callback);
    }

    /**
     * 获取tokens列表
     * @returns {Array}
     */
    getTokens() {
        return this.analysisResult ? this.analysisResult.tokens : [];
    }

    /**
     * 获取token数量
     * @returns {number}
     */
    getTokenCount() {
        return this.getTokens().length;
    }

    /**
     * 检查是否有分析结果
     * @returns {boolean}
     */
    hasAnalysisResult() {
        return this.analysisResult !== null;
    }

    /**
     * 检查是否已加载模型
     * @returns {boolean}
     */
    hasModel() {
        return this.currentModel !== null;
    }

    /**
     * 获取分析摘要
     * @returns {Object|null}
     */
    getAnalysisSummary() {
        return this.analysisResult ? this.analysisResult.analysis_summary : null;
    }

    /**
     * 获取token分析结果
     * @returns {Object|null}
     */
    getTokenAnalysis() {
        return this.analysisResult ? this.analysisResult.token_analysis : null;
    }
}

// 创建全局状态管理器
window.modelState = new ModelState();

// API错误处理器
class ErrorHandler {
    constructor() {
        this.errorContainer = null;
    }

    /**
     * 显示错误消息
     * @param {string|Error} error 错误信息
     * @param {string} context 错误上下文
     */
    showError(error, context = '') {
        const message = error instanceof Error ? error.message : error;
        const fullMessage = context ? `${context}: ${message}` : message;
        
        console.error('Application Error:', fullMessage);
        
        // 显示用户友好的错误消息
        this.displayUserError(fullMessage);
        
        // 发送错误事件
        window.modelState.eventEmitter.emit('error', {
            message: fullMessage,
            context,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * 显示用户友好的错误消息
     * @param {string} message 错误消息
     */
    displayUserError(message) {
        // 创建或更新错误提示
        let errorDiv = document.getElementById('error-notification');
        
        if (!errorDiv) {
            errorDiv = Utils.createElement('div', {
                id: 'error-notification',
                className: 'error-notification'
            });
            document.body.appendChild(errorDiv);
        }
        
        errorDiv.innerHTML = `
            <div class="error-content">
                <i class="fas fa-exclamation-triangle"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        errorDiv.style.display = 'block';
        
        // 自动隐藏
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    }

    /**
     * 清除所有错误消息
     */
    clearErrors() {
        const errorDiv = document.getElementById('error-notification');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
}

// 创建全局错误处理器
window.errorHandler = new ErrorHandler();

// 添加错误通知的CSS（动态注入）
const errorNotificationCSS = `
.error-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #fed7d7;
    border: 2px solid #fc8181;
    border-radius: 8px;
    padding: 0;
    max-width: 400px;
    z-index: 3000;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    animation: slideInRight 0.3s ease-out;
    display: none;
}

.error-content {
    padding: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #c53030;
}

.error-content i:first-child {
    color: #e53e3e;
    font-size: 18px;
}

.error-content span {
    flex: 1;
    font-weight: 500;
}

.error-content button {
    background: none;
    border: none;
    color: #c53030;
    cursor: pointer;
    padding: 5px;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.error-content button:hover {
    background: rgba(197, 48, 48, 0.1);
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}
`;

// 注入CSS
const style = document.createElement('style');
style.textContent = errorNotificationCSS;
document.head.appendChild(style);
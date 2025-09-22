// 工具函数集合

/**
 * 显示加载状态
 * @param {string} text 加载文本
 */
function showLoading(text = '加载中...') {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    loadingText.textContent = text;
    overlay.classList.add('show');
}

/**
 * 隐藏加载状态
 */
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.remove('show');
}

/**
 * 显示状态信息
 * @param {HTMLElement} element 状态元素
 * @param {string} message 消息内容
 * @param {string} type 状态类型: success, error, loading
 */
function showStatus(element, message, type = 'success') {
    element.textContent = message;
    element.className = `status-info status-${type}`;
    element.style.display = 'block';
}

/**
 * 清空状态信息
 * @param {HTMLElement} element 状态元素
 */
function clearStatus(element) {
    element.textContent = '';
    element.className = 'status-info';
    element.style.display = 'none';
}

/**
 * 防抖函数
 * @param {Function} func 要防抖的函数
 * @param {number} wait 等待时间(ms)
 * @param {boolean} immediate 是否立即执行
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * 节流函数
 * @param {Function} func 要节流的函数
 * @param {number} limit 时间限制(ms)
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

/**
 * 创建元素的便捷函数
 * @param {string} tag 标签名
 * @param {Object} attributes 属性对象
 * @param {string|Node[]} content 内容
 * @returns {HTMLElement}
 */
function createElement(tag, attributes = {}, content = '') {
    const element = document.createElement(tag);
    
    // 设置属性
    Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'innerHTML') {
            element.innerHTML = value;
        } else if (key.startsWith('data-')) {
            element.setAttribute(key, value);
        } else {
            element[key] = value;
        }
    });
    
    // 设置内容
    if (typeof content === 'string') {
        element.textContent = content;
    } else if (Array.isArray(content)) {
        content.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else {
                element.appendChild(child);
            }
        });
    } else if (content instanceof Node) {
        element.appendChild(content);
    }
    
    return element;
}

/**
 * 格式化数字
 * @param {number} num 数字
 * @param {number} decimals 小数位数
 * @returns {string}
 */
function formatNumber(num, decimals = 3) {
    if (typeof num !== 'number' || isNaN(num)) return 'N/A';
    return num.toFixed(decimals);
}

/**
 * 格式化大数字(K, M, B)
 * @param {number} num 数字
 * @returns {string}
 */
function formatLargeNumber(num) {
    if (typeof num !== 'number' || isNaN(num)) return 'N/A';
    
    const absNum = Math.abs(num);
    if (absNum >= 1e9) {
        return (num / 1e9).toFixed(1) + 'B';
    } else if (absNum >= 1e6) {
        return (num / 1e6).toFixed(1) + 'M';
    } else if (absNum >= 1e3) {
        return (num / 1e3).toFixed(1) + 'K';
    } else {
        return num.toString();
    }
}

/**
 * 生成随机颜色
 * @param {number} alpha 透明度
 * @returns {string}
 */
function getRandomColor(alpha = 1) {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * HSL颜色生成器
 * @param {number} index 索引
 * @param {number} total 总数
 * @param {number} saturation 饱和度 (0-1)
 * @param {number} lightness 明度 (0-1)
 * @returns {string}
 */
function getHSLColor(index, total, saturation = 0.7, lightness = 0.5) {
    const hue = (index * 360) / total;
    return `hsl(${hue}, ${saturation * 100}%, ${lightness * 100}%)`;
}

/**
 * 计算颜色强度映射
 * @param {number} value 值
 * @param {number} min 最小值
 * @param {number} max 最大值
 * @returns {string}
 */
function getIntensityColor(value, min, max) {
    if (max === min) return 'rgba(102, 126, 234, 0.5)';
    
    const normalized = (value - min) / (max - min);
    const intensity = Math.max(0, Math.min(1, normalized));
    
    // 从浅蓝到深蓝的渐变
    const r = Math.floor(102 + (255 - 102) * (1 - intensity));
    const g = Math.floor(126 + (255 - 126) * (1 - intensity));
    const b = 234;
    
    return `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.7})`;
}

/**
 * 数组排序的比较函数
 * @param {string} key 排序键
 * @param {boolean} reverse 是否倒序
 */
function compareBy(key, reverse = false) {
    return (a, b) => {
        const aVal = a[key];
        const bVal = b[key];
        
        if (aVal < bVal) return reverse ? 1 : -1;
        if (aVal > bVal) return reverse ? -1 : 1;
        return 0;
    };
}

/**
 * 深拷贝对象
 * @param {*} obj 要拷贝的对象
 * @returns {*}
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj.getTime());
    if (obj instanceof Array) return obj.map(item => deepClone(item));
    if (typeof obj === 'object') {
        const cloned = {};
        Object.keys(obj).forEach(key => {
            cloned[key] = deepClone(obj[key]);
        });
        return cloned;
    }
}

/**
 * 获取元素的绝对位置
 * @param {HTMLElement} element 元素
 * @returns {Object}
 */
function getElementPosition(element) {
    const rect = element.getBoundingClientRect();
    return {
        top: rect.top + window.scrollY,
        left: rect.left + window.scrollX,
        width: rect.width,
        height: rect.height,
        centerX: rect.left + rect.width / 2,
        centerY: rect.top + rect.height / 2
    };
}

/**
 * 创建SVG元素
 * @param {string} tag SVG标签名
 * @param {Object} attributes 属性
 * @returns {SVGElement}
 */
function createSVGElement(tag, attributes = {}) {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => {
        element.setAttribute(key, value);
    });
    return element;
}

/**
 * 动画帧请求的便捷包装
 * @param {Function} callback 回调函数
 * @param {number} duration 持续时间(ms)
 */
function animate(callback, duration = 1000) {
    const startTime = performance.now();
    
    function frame(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        callback(progress);
        
        if (progress < 1) {
            requestAnimationFrame(frame);
        }
    }
    
    requestAnimationFrame(frame);
}

/**
 * 缓动函数
 */
const Easing = {
    easeInOut: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeIn: t => t * t,
    easeOut: t => t * (2 - t),
    elastic: t => {
        if (t === 0 || t === 1) return t;
        const p = 0.3;
        const s = p / 4;
        return Math.pow(2, -10 * t) * Math.sin((t - s) * (2 * Math.PI) / p) + 1;
    }
};

/**
 * 本地存储助手
 */
const Storage = {
    get: (key, defaultValue = null) => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.warn('Failed to get from localStorage:', e);
            return defaultValue;
        }
    },
    
    set: (key, value) => {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.warn('Failed to set to localStorage:', e);
            return false;
        }
    },
    
    remove: (key) => {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.warn('Failed to remove from localStorage:', e);
            return false;
        }
    }
};

/**
 * 事件发射器
 */
class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }
    
    off(event, callback) {
        if (!this.events[event]) return;
        this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
    
    emit(event, ...args) {
        if (!this.events[event]) return;
        this.events[event].forEach(callback => callback(...args));
    }
}

// 导出到全局作用域
window.Utils = {
    showLoading,
    hideLoading,
    showStatus,
    clearStatus,
    debounce,
    throttle,
    createElement,
    formatNumber,
    formatLargeNumber,
    getRandomColor,
    getHSLColor,
    getIntensityColor,
    compareBy,
    deepClone,
    getElementPosition,
    createSVGElement,
    animate,
    Easing,
    Storage,
    EventEmitter
};
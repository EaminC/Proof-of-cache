// 可视化核心模块

class TokenVisualizer {
    constructor(container) {
        this.container = container;
        this.tokens = [];
        this.selectedTokenIndex = null;
        this.highlightedTokens = new Set();
    }

    /**
     * 渲染token序列
     * @param {Array} tokens token列表
     */
    renderTokens(tokens) {
        this.tokens = tokens;
        this.container.innerHTML = '';
        
        if (!tokens || tokens.length === 0) {
            this.container.innerHTML = '<p class="placeholder">暂无tokens</p>';
            return;
        }

        tokens.forEach((token, index) => {
            const tokenElement = this.createTokenElement(token, index);
            this.container.appendChild(tokenElement);
        });

        // 添加淡入动画
        this.container.classList.add('fade-in');
    }

    /**
     * 创建token元素
     * @param {string} token token文本
     * @param {number} index 索引
     * @returns {HTMLElement}
     */
    createTokenElement(token, index) {
        const tokenDiv = Utils.createElement('div', {
            className: 'token',
            'data-index': index,
            title: `Token ${index}: "${token}"`
        });

        // Token文本
        const tokenText = Utils.createElement('span', {
            className: 'token-text'
        }, token);

        // Token索引
        const tokenIndex = Utils.createElement('span', {
            className: 'token-index'
        }, index.toString());

        tokenDiv.appendChild(tokenText);
        tokenDiv.appendChild(tokenIndex);

        // 点击事件
        tokenDiv.addEventListener('click', () => {
            this.selectToken(index);
        });

        // 悬停效果
        tokenDiv.addEventListener('mouseenter', () => {
            this.highlightToken(index, true);
        });

        tokenDiv.addEventListener('mouseleave', () => {
            this.highlightToken(index, false);
        });

        return tokenDiv;
    }

    /**
     * 选择token
     * @param {number} index token索引
     */
    selectToken(index) {
        // 移除之前的选中状态
        if (this.selectedTokenIndex !== null) {
            const prevSelected = this.container.querySelector(`[data-index="${this.selectedTokenIndex}"]`);
            if (prevSelected) {
                prevSelected.classList.remove('selected');
            }
        }

        // 设置新的选中状态
        this.selectedTokenIndex = index;
        const tokenElement = this.container.querySelector(`[data-index="${index}"]`);
        if (tokenElement) {
            tokenElement.classList.add('selected');
            
            // 滚动到可视区域
            tokenElement.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest',
                inline: 'center'
            });
        }

        // 更新状态管理器
        window.modelState.selectToken(index);

        // 更新UI显示
        this.updateSelectedTokenInfo(index);
    }

    /**
     * 高亮token
     * @param {number} index token索引
     * @param {boolean} highlight 是否高亮
     */
    highlightToken(index, highlight) {
        const tokenElement = this.container.querySelector(`[data-index="${index}"]`);
        if (tokenElement) {
            if (highlight) {
                tokenElement.classList.add('highlighted');
            } else {
                tokenElement.classList.remove('highlighted');
            }
        }
    }

    /**
     * 批量高亮tokens
     * @param {Array} indices token索引数组
     */
    highlightTokens(indices) {
        // 清除之前的高亮
        this.clearHighlights();
        
        // 添加新的高亮
        indices.forEach(index => {
            this.highlightedTokens.add(index);
            this.highlightToken(index, true);
        });
    }

    /**
     * 清除所有高亮
     */
    clearHighlights() {
        this.highlightedTokens.forEach(index => {
            this.highlightToken(index, false);
        });
        this.highlightedTokens.clear();
    }

    /**
     * 更新选中token信息
     * @param {number} index token索引
     */
    updateSelectedTokenInfo(index) {
        const selectedTokenSpan = document.getElementById('selected-token');
        if (selectedTokenSpan && this.tokens[index]) {
            selectedTokenSpan.textContent = `选中token: ${index} - "${this.tokens[index]}"`;
        }
    }
}

class ModelStructureVisualizer {
    constructor(container) {
        this.container = container;
        this.structure = null;
        this.activeLayer = null;
    }

    /**
     * 渲染模型结构
     * @param {Object} structure 模型结构数据
     */
    renderStructure(structure) {
        this.structure = structure;
        this.container.innerHTML = '';

        if (!structure) {
            this.container.innerHTML = '<p class="placeholder">暂无模型结构</p>';
            return;
        }

        // 创建模型信息头部
        const header = this.createStructureHeader(structure);
        this.container.appendChild(header);

        // 创建层级列表
        const layersContainer = Utils.createElement('div', {
            className: 'layers-container'
        });

        structure.layers.forEach((layer, index) => {
            const layerElement = this.createLayerElement(layer, index);
            layersContainer.appendChild(layerElement);
        });

        this.container.appendChild(layersContainer);
    }

    /**
     * 创建结构头部
     * @param {Object} structure 结构数据
     * @returns {HTMLElement}
     */
    createStructureHeader(structure) {
        return Utils.createElement('div', {
            className: 'structure-header'
        }, [
            Utils.createElement('h4', {}, `${structure.model_type.toUpperCase()}`),
            Utils.createElement('div', {
                className: 'structure-stats'
            }, [
                Utils.createElement('span', {}, `${structure.num_layers} 层`),
                Utils.createElement('span', {}, `${structure.num_heads} 头`),
                Utils.createElement('span', {}, `${structure.hidden_size}D`)
            ])
        ]);
    }

    /**
     * 创建层级元素
     * @param {Object} layer 层级数据
     * @param {number} index 索引
     * @returns {HTMLElement}
     */
    createLayerElement(layer, index) {
        const layerDiv = Utils.createElement('div', {
            className: 'layer-node',
            'data-layer': index
        });

        // 层标题
        const title = Utils.createElement('div', {
            className: 'layer-title'
        }, [
            Utils.createElement('span', {}, `Layer ${index}`),
            Utils.createElement('span', {
                className: 'layer-status'
            }, '○')
        ]);

        // 组件列表
        const components = Utils.createElement('div', {
            className: 'layer-components'
        });

        layer.components.forEach(component => {
            const compElement = Utils.createElement('span', {
                className: 'component',
                title: this.getComponentDescription(component)
            }, component.name);
            components.appendChild(compElement);
        });

        layerDiv.appendChild(title);
        layerDiv.appendChild(components);

        // 点击事件
        layerDiv.addEventListener('click', () => {
            this.selectLayer(index);
        });

        return layerDiv;
    }

    /**
     * 选择层级
     * @param {number} layerIndex 层级索引
     */
    selectLayer(layerIndex) {
        // 移除之前的选中状态
        if (this.activeLayer !== null) {
            const prevActive = this.container.querySelector(`[data-layer="${this.activeLayer}"]`);
            if (prevActive) {
                prevActive.classList.remove('active');
            }
        }

        // 设置新的选中状态
        this.activeLayer = layerIndex;
        const layerElement = this.container.querySelector(`[data-layer="${layerIndex}"]`);
        if (layerElement) {
            layerElement.classList.add('active');
        }

        // 触发层级选择事件
        window.modelState.eventEmitter.emit('layerSelected', layerIndex);
    }

    /**
     * 高亮特定层级
     * @param {Array} layerIndices 层级索引数组
     */
    highlightLayers(layerIndices) {
        // 清除之前的高亮
        this.container.querySelectorAll('.layer-node').forEach(node => {
            node.classList.remove('highlighted');
        });

        // 添加新的高亮
        layerIndices.forEach(index => {
            const layerElement = this.container.querySelector(`[data-layer="${index}"]`);
            if (layerElement) {
                layerElement.classList.add('highlighted');
            }
        });
    }

    /**
     * 获取组件描述
     * @param {Object} component 组件数据
     * @returns {string}
     */
    getComponentDescription(component) {
        switch (component.type) {
            case 'multi_head_attention':
                return `多头注意力 (${component.num_heads} heads)`;
            case 'feedforward':
                return `前馈网络 (${Utils.formatLargeNumber(component.hidden_size)})`;
            case 'layer_norm':
                return '层归一化';
            default:
                return component.name;
        }
    }
}

class AttentionHeatmapVisualizer {
    constructor(container) {
        this.container = container;
        this.attentionData = null;
        this.currentLayer = 0;
        this.currentHead = 0;
        this.tokens = [];
    }

    /**
     * 设置注意力数据
     * @param {Object} attentionData 注意力数据
     * @param {Array} tokens token列表
     */
    setData(attentionData, tokens) {
        this.attentionData = attentionData;
        this.tokens = tokens;
        this.renderHeatmap();
    }

    /**
     * 设置当前层和头
     * @param {number} layer 层索引
     * @param {number} head 头索引
     */
    setLayerAndHead(layer, head) {
        this.currentLayer = layer;
        this.currentHead = head;
        this.renderHeatmap();
    }

    /**
     * 渲染热图
     */
    renderHeatmap() {
        if (!this.attentionData || !this.tokens.length) {
            this.container.innerHTML = '<p class="placeholder">暂无注意力数据</p>';
            return;
        }

        // 获取当前层和头的注意力权重
        const layerKey = `layer_${this.currentLayer}`;
        const layerData = this.attentionData[layerKey];
        
        if (!layerData || !layerData.weights) {
            this.container.innerHTML = '<p class="placeholder">当前层暂无注意力数据</p>';
            return;
        }

        const weights = layerData.weights[this.currentHead] || layerData.weights[0];
        if (!weights) {
            this.container.innerHTML = '<p class="placeholder">当前头暂无注意力数据</p>';
            return;
        }

        // 使用Plotly渲染热图
        this.renderPlotlyHeatmap(weights);
    }

    /**
     * 使用Plotly渲染热图
     * @param {Array} weights 注意力权重矩阵
     */
    renderPlotlyHeatmap(weights) {
        const data = [{
            z: weights,
            type: 'heatmap',
            colorscale: [
                [0, '#f7fafc'],
                [0.5, '#667eea'],
                [1, '#2d3748']
            ],
            showscale: true,
            hoverongaps: false,
            hovertemplate: 
                '<b>From:</b> %{y}<br>' +
                '<b>To:</b> %{x}<br>' +
                '<b>Attention:</b> %{z:.4f}<br>' +
                '<extra></extra>'
        }];

        const layout = {
            title: {
                text: `注意力热图 - Layer ${this.currentLayer}, Head ${this.currentHead}`,
                font: { size: 14 }
            },
            xaxis: {
                title: 'To Token',
                tickvals: this.tokens.map((_, i) => i),
                ticktext: this.tokens.map((token, i) => `${i}: ${token.slice(0, 8)}${token.length > 8 ? '...' : ''}`),
                tickangle: -45
            },
            yaxis: {
                title: 'From Token',
                tickvals: this.tokens.map((_, i) => i),
                ticktext: this.tokens.map((token, i) => `${i}: ${token.slice(0, 8)}${token.length > 8 ? '...' : ''}`),
                autorange: 'reversed'
            },
            width: this.container.offsetWidth - 20,
            height: 300,
            margin: { l: 100, r: 20, t: 50, b: 100 }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(this.container, data, layout, config);

        // 点击事件
        this.container.on('plotly_click', (data) => {
            const point = data.points[0];
            const fromToken = point.y;
            const toToken = point.x;
            
            // 高亮相关tokens
            window.tokenVisualizer.highlightTokens([fromToken, toToken]);
            
            // 显示注意力详情
            this.showAttentionDetails(fromToken, toToken, point.z);
        });
    }

    /**
     * 显示注意力详情
     * @param {number} fromToken 源token
     * @param {number} toToken 目标token
     * @param {number} weight 注意力权重
     */
    showAttentionDetails(fromToken, toToken, weight) {
        const details = `
            <div class="attention-detail">
                <h4>注意力连接</h4>
                <p><strong>从:</strong> ${fromToken} "${this.tokens[fromToken]}"</p>
                <p><strong>到:</strong> ${toToken} "${this.tokens[toToken]}"</p>
                <p><strong>权重:</strong> ${Utils.formatNumber(weight, 4)}</p>
                <p><strong>层级:</strong> ${this.currentLayer}</p>
                <p><strong>头部:</strong> ${this.currentHead}</p>
            </div>
        `;
        
        // 显示在侧边面板
        this.showInSidePanel(details);
    }

    /**
     * 在侧边面板显示内容
     * @param {string} content HTML内容
     */
    showInSidePanel(content) {
        const sidePanel = document.getElementById('side-panel');
        const sidePanelContent = document.getElementById('side-panel-content');
        
        sidePanelContent.innerHTML = content;
        sidePanel.classList.add('open');
    }
}

class TokenDetailVisualizer {
    constructor(container) {
        this.container = container;
        this.currentTokenDetails = null;
    }

    /**
     * 显示token详情
     * @param {Object} details token详情数据
     */
    showDetails(details) {
        this.currentTokenDetails = details;
        this.container.innerHTML = '';

        if (!details) {
            this.container.innerHTML = '<p class="placeholder">点击token查看详细信息</p>';
            return;
        }

        // 基本信息
        const basicInfo = this.createBasicInfo(details);
        this.container.appendChild(basicInfo);

        // 层级分析
        const layerAnalysis = this.createLayerAnalysis(details.layer_data);
        this.container.appendChild(layerAnalysis);

        // 注意力流
        if (details.attention_flow) {
            const attentionFlow = this.createAttentionFlow(details.attention_flow);
            this.container.appendChild(attentionFlow);
        }
    }

    /**
     * 创建基本信息部分
     * @param {Object} details 详情数据
     * @returns {HTMLElement}
     */
    createBasicInfo(details) {
        return Utils.createElement('div', {
            className: 'detail-section'
        }, [
            Utils.createElement('div', {
                className: 'detail-title'
            }, [
                Utils.createElement('i', {
                    className: 'fas fa-info-circle'
                }),
                Utils.createElement('span', {}, '基本信息')
            ]),
            Utils.createElement('div', {
                className: 'detail-content'
            }, [
                Utils.createElement('div', {
                    className: 'metric'
                }, [
                    Utils.createElement('span', {}, 'Token文本:'),
                    Utils.createElement('span', {
                        className: 'metric-value'
                    }, `"${details.token_text}"`)
                ]),
                Utils.createElement('div', {
                    className: 'metric'
                }, [
                    Utils.createElement('span', {}, '位置:'),
                    Utils.createElement('span', {
                        className: 'metric-value'
                    }, details.token_position.toString())
                ]),
                Utils.createElement('div', {
                    className: 'metric'
                }, [
                    Utils.createElement('span', {}, 'Token ID:'),
                    Utils.createElement('span', {
                        className: 'metric-value'
                    }, details.token_id.toString())
                ])
            ])
        ]);
    }

    /**
     * 创建层级分析部分
     * @param {Object} layerData 层级数据
     * @returns {HTMLElement}
     */
    createLayerAnalysis(layerData) {
        const container = Utils.createElement('div', {
            className: 'detail-section'
        });

        const title = Utils.createElement('div', {
            className: 'detail-title'
        }, [
            Utils.createElement('i', {
                className: 'fas fa-layers'
            }),
            Utils.createElement('span', {}, '层级分析')
        ]);

        const content = Utils.createElement('div', {
            className: 'detail-content'
        });

        // 为每层创建信息
        Object.entries(layerData).forEach(([layerIndex, layerInfo]) => {
            const layerDiv = Utils.createElement('div', {
                className: 'layer-detail'
            });

            const layerTitle = Utils.createElement('h5', {}, `Layer ${layerIndex}`);
            layerDiv.appendChild(layerTitle);

            if (layerInfo.hidden_state_norm !== null) {
                const normMetric = Utils.createElement('div', {
                    className: 'metric'
                }, [
                    Utils.createElement('span', {}, '隐藏状态范数:'),
                    Utils.createElement('span', {
                        className: 'metric-value'
                    }, Utils.formatNumber(layerInfo.hidden_state_norm))
                ]);
                layerDiv.appendChild(normMetric);
            }

            if (layerInfo.attention_weights) {
                const attentionDiv = Utils.createElement('div', {}, '注意力权重:');
                Object.entries(layerInfo.attention_weights).forEach(([headIndex, headData]) => {
                    const headInfo = Utils.createElement('div', {
                        className: 'metric small'
                    }, [
                        Utils.createElement('span', {}, `Head ${headIndex}:`),
                        Utils.createElement('span', {
                            className: 'metric-value'
                        }, Utils.formatNumber(headData.norm))
                    ]);
                    attentionDiv.appendChild(headInfo);
                });
                layerDiv.appendChild(attentionDiv);
            }

            content.appendChild(layerDiv);
        });

        container.appendChild(title);
        container.appendChild(content);
        return container;
    }

    /**
     * 创建注意力流部分
     * @param {Object} attentionFlow 注意力流数据
     * @returns {HTMLElement}
     */
    createAttentionFlow(attentionFlow) {
        return Utils.createElement('div', {
            className: 'detail-section'
        }, [
            Utils.createElement('div', {
                className: 'detail-title'
            }, [
                Utils.createElement('i', {
                    className: 'fas fa-eye'
                }),
                Utils.createElement('span', {}, '注意力流')
            ]),
            Utils.createElement('div', {
                className: 'detail-content'
            }, [
                Utils.createElement('p', {}, '出向注意力: ' + JSON.stringify(attentionFlow.outgoing_attention || {}, null, 2)),
                Utils.createElement('p', {}, '入向注意力: ' + JSON.stringify(attentionFlow.incoming_attention || {}, null, 2))
            ])
        ]);
    }
}

// 状态轨迹图表可视化
class StateTrajectoryVisualizer {
    constructor(container) {
        this.container = container;
        this.trajectoryData = null;
    }

    /**
     * 显示状态轨迹
     * @param {Object} tokenDetails token详情数据
     */
    showTrajectory(tokenDetails) {
        if (!tokenDetails || !tokenDetails.layer_data) {
            this.container.innerHTML = '<p class="placeholder">选择token查看状态轨迹</p>';
            return;
        }

        this.renderTrajectoryChart(tokenDetails);
    }

    /**
     * 渲染轨迹图表
     * @param {Object} tokenDetails token详情数据
     */
    renderTrajectoryChart(tokenDetails) {
        const layers = Object.keys(tokenDetails.layer_data).map(Number).sort((a, b) => a - b);
        const norms = layers.map(layer => tokenDetails.layer_data[layer].hidden_state_norm || 0);

        const data = [{
            x: layers,
            y: norms,
            type: 'scatter',
            mode: 'lines+markers',
            name: `Token "${tokenDetails.token_text}"`,
            line: {
                color: '#667eea',
                width: 3
            },
            marker: {
                color: '#667eea',
                size: 8
            },
            hovertemplate: 
                '<b>Layer:</b> %{x}<br>' +
                '<b>Norm:</b> %{y:.4f}<br>' +
                '<extra></extra>'
        }];

        const layout = {
            title: {
                text: `状态轨迹 - Token ${tokenDetails.token_position}: "${tokenDetails.token_text}"`,
                font: { size: 14 }
            },
            xaxis: {
                title: '层级',
                dtick: 1
            },
            yaxis: {
                title: '隐藏状态范数'
            },
            width: this.container.offsetWidth - 20,
            height: 300,
            margin: { l: 60, r: 20, t: 50, b: 50 }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(this.container, data, layout, config);
    }
}

// 导出到全局作用域
window.TokenVisualizer = TokenVisualizer;
window.ModelStructureVisualizer = ModelStructureVisualizer;
window.AttentionHeatmapVisualizer = AttentionHeatmapVisualizer;
window.TokenDetailVisualizer = TokenDetailVisualizer;
window.StateTrajectoryVisualizer = StateTrajectoryVisualizer;
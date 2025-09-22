# 🌐 外部访问配置指南

## 当前服务器状态

**✅ 服务已启动并可外部访问！**

- **本地访问**: http://localhost:5000  
- **外部访问**: http://192.5.86.157:5000
- **内网访问**: http://10.140.82.102:5000
- **服务状态**: 运行中，监听所有网络接口

## 🔗 访问方式

### 1. 外部互联网访问
从任何有互联网连接的设备访问：
```
http://192.5.86.157:5000
```

### 2. 同一局域网访问
如果您在同一个内网中，也可以使用：
```
http://10.140.82.102:5000
```

### 3. 移动设备访问
在手机或平板上打开浏览器，输入：
```
http://192.5.86.157:5000
```

## ⚠️ 常见访问问题

### 问题1: 无法连接到服务器
**可能原因**: 防火墙阻止了端口5000

**解决方案**:
```bash
# Ubuntu/Debian系统
sudo ufw allow 5000
sudo ufw reload

# CentOS/RHEL系统  
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload

# 检查端口是否开放
sudo netstat -tlnp | grep :5000
```

### 问题2: 连接被拒绝
**可能原因**: 服务器未正确启动

**解决方案**:
```bash
# 检查服务是否运行
ps aux | grep web_visualizer

# 重新启动服务
cd /home/cc/poc
python3 web_visualizer.py
```

### 问题3: 页面加载缓慢
**可能原因**: 网络延迟或服务器资源不足

**解决方案**:
- 确保服务器有足够内存
- 使用较小的模型进行测试
- 减少同时分析的token数量

## 🚀 快速测试

### 测试连接
```bash
# 从其他机器测试外部连接
curl http://192.5.86.157:5000

# 测试内网连接
curl http://10.140.82.102:5000
```

### 测试API
```bash
# 测试健康检查
curl http://192.5.86.157:5000/api/health
```

## 🔧 高级配置

### 自定义端口
如果需要使用其他端口：
```bash
export PORT=8080
python3 web_visualizer.py
```

### 生产环境部署
对于生产环境，建议使用Gunicorn：
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_visualizer:app
```

### HTTPS配置
如果需要HTTPS访问：
```bash
# 生成自签名证书（仅用于测试）
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# 修改启动方式
app.run(host='0.0.0.0', port=5000, debug=True, 
        ssl_context=('cert.pem', 'key.pem'))
```

## 📱 移动端优化

Web界面已针对移动端进行优化：
- 响应式布局
- 触摸友好的交互
- 适配小屏幕显示

## 🛡️ 安全注意事项

### 开发环境
当前配置适用于开发和测试：
- Debug模式开启
- 详细错误信息显示
- 热重载功能

### 生产环境建议
如果用于生产，请：
1. 关闭Debug模式
2. 配置适当的认证
3. 使用HTTPS
4. 设置防火墙规则
5. 监控资源使用

## 🎯 使用示例

现在您可以：

1. **在浏览器中打开**: http://192.5.86.157:5000
2. **选择模型**: 点击"加载模型"选择GPT-2
3. **输入文本**: 在文本框中输入 "Hello transformer world"
4. **开始分析**: 点击"开始分析"按钮
5. **交互探索**: 点击任意token查看中间状态

## 📊 网络监控

查看服务器网络状态：
```bash
# 查看监听端口
sudo netstat -tlnp | grep python

# 查看连接状态
sudo netstat -an | grep :5000

# 实时监控连接
watch -n 1 "netstat -an | grep :5000"
```

## 🔄 服务管理

### 启动服务
```bash
cd /home/cc/poc
python3 web_visualizer.py
```

### 停止服务
```bash
# 在运行的终端中按 Ctrl+C
# 或者强制停止
pkill -f "web_visualizer.py"
```

### 后台运行
```bash
# 使用nohup在后台运行
nohup python3 web_visualizer.py > visualizer.log 2>&1 &

# 查看日志
tail -f visualizer.log
```

---

**🎉 现在您的Transformer可视化工具已经可以从任何设备访问了！**

在浏览器中打开 **http://192.5.86.157:5000** 开始探索吧！

## 🔥 重要提示

**主要访问地址**: http://192.5.86.157:5000

这个地址可以从世界任何地方访问，只要有互联网连接！

**内网备用地址**: http://10.140.82.102:5000 (仅限同一局域网)
import requests
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

class IntentRecognizer:
    """意图识别器类 - 连接DeepSeek V3模型进行意图识别"""

    def __init__(self):
        # 数据库查询关键词
        self.database_keywords = {
            '交易流水表': ['交易流水', '交易记录', 'transaction_id', '交易金额', '交易时间'],
            '机构信息表': ['机构信息', 'institution_id', '机构名称', '许可证号'],
            '商户信息表': ['商户信息', 'merchant_id', '商户名称', '营业执照']
        }
        
        # API工具映射表
        self.api_tools_mapping = {
            '信用卡服务': {
                'keywords': ['信用卡', '账单', 'cardNumber', '月账单', '信用卡账单'],
                'required_params': ['cardNumber', 'month'],
                'optional_params': []
            },
            '汇率服务': {
                'keywords': ['汇率', '兑换', '货币', '等于多少', '换成', '人民币', '美元', '日元', '韩元'],
                'required_params': ['fromCurrency', 'toCurrency'],
                'optional_params': ['amount']
            },
            '水电煤服务': {
                'keywords': ['水电煤', '电费', '水费', '煤气费', '户号', 'householdId', '用电量', '用水量'],
                'required_params': ['householdId', 'month'],
                'optional_params': ['utilityType']
            },
            '用户资产服务': {
                'keywords': ['用户资产', '资产信息', 'customerId', '资产查询'],
                'required_params': ['customerId'],
                'optional_params': ['assetType']
            },
            '支付订单服务': {
                'keywords': ['支付订单', '创建订单', 'merchantId', 'orderId', '支付'],
                'required_params': ['merchantId', 'orderId'],
                'optional_params': ['amount']
            },
            '获取当前日期工具': {
                'keywords': ['当前日期', '现在时间', '今天日期', '现在几点', '今天几号'],
                'required_params': [],
                'optional_params': []
            },
            '计算器工具': {
                'keywords': ['计算', '等于多少', '算式', '表达式', '算一下', '计算结果'],
                'required_params': ['expression'],
                'optional_params': []
            }
        }

    def extract_parameters(self, question: str, tool_name: str) -> dict[str, any]:
        """从问题中提取参数"""
        params = {}

        if tool_name == '信用卡服务':
            # 提取卡号和月份
            card_match = re.search(r'[0-9]{16}', question)
            month_match = re.search(r'20[0-9]{2}-[0-9]{2}', question)
            if card_match:
                params['cardNumber'] = card_match.group()
            if month_match:
                params['month'] = month_match.group()
            else:
                # 默认当前月份
                current_month = datetime.now().strftime('%Y-%m')
                params['month'] = current_month

        elif tool_name == '汇率服务':
            # 改进汇率参数提取
            currency_mapping = {
                '人民币': 'CNY', '元': 'CNY', '人民币元': 'CNY',
                '美元': 'USD', '美金': 'USD', '美圆': 'USD',
                '日元': 'JPY', '円': 'JPY', '日圆': 'JPY',
                '韩元': 'KRW', '韩圜': 'KRW', '韩圆': 'KRW', '韩币': 'KRW',
                '欧元': 'EUR', '欧圆': 'EUR',
                '英镑': 'GBP', '英磅': 'GBP'
            }
            
            # 提取金额
            amount_match = re.search(r'(\d+(?:\.\d+)?)', question)
            if amount_match:
                params['amount'] = float(amount_match.group(1))
            
            # 提取货币对 - 按出现顺序提取
            found_currencies = []
            for chinese_name, code in currency_mapping.items():
                if chinese_name in question:
                    # 记录货币名称和位置
                    pos = question.find(chinese_name)
                    found_currencies.append((pos, code, chinese_name))
            
            # 按位置排序
            found_currencies.sort(key=lambda x: x[0])
            
            if len(found_currencies) >= 2:
                params['fromCurrency'] = found_currencies[0][1]  # 第一个出现的货币
                params['toCurrency'] = found_currencies[1][1]     # 第二个出现的货币
            elif len(found_currencies) == 1:
                # 根据问题判断方向
                currency_name = found_currencies[0][2]
                if '等于多少' in question or '兑换' in question:
                    # 如果是"等于多少"格式，第一个货币是源货币
                    params['fromCurrency'] = found_currencies[0][1]
                    params['toCurrency'] = 'CNY'  # 默认转换为人民币
                else:
                    # 其他情况默认从该货币转换为人民币
                    params['fromCurrency'] = found_currencies[0][1]
                    params['toCurrency'] = 'CNY'

        elif tool_name == '水电煤服务':
            # 提取户号、月份和类型
            household_match = re.search(r'[A-Z]{2}\d+', question)
            month_match = re.search(r'20[0-9]{2}-[0-9]{2}', question)

            if household_match:
                params['householdId'] = household_match.group()
            if month_match:
                params['month'] = month_match.group()
            else:
                # 默认当前月份
                current_month = datetime.now().strftime('%Y-%m')
                params['month'] = current_month

            if '电' in question:
                params['utilityType'] = 'electricity'
            elif '水' in question:
                params['utilityType'] = 'water'
            elif '煤' in question or '气' in question:
                params['utilityType'] = 'gas'
            else:
                params['utilityType'] = 'electricity'  # 默认电费

        elif tool_name == '用户资产服务':
            # 提取用户ID
            id_match = re.search(r'[0-9]{18}|[0-9]{15}', question)
            if id_match:
                params['customerId'] = id_match.group()
            
            if '房产' in question or '房子' in question:
                params['assetType'] = 'household'
            else:
                params['assetType'] = 'card'  # 默认信用卡资产

        elif tool_name == '支付订单服务':
            # 提取商户ID和订单ID
            merchant_match = re.search(r'M\d+', question, re.IGNORECASE)
            order_match = re.search(r'ORD\d+', question, re.IGNORECASE)
            amount_match = re.search(r'(\d+(?:\.\d+)?)元', question)

            if merchant_match:
                params['merchantId'] = merchant_match.group()
            if order_match:
                params['orderId'] = order_match.group()
            if amount_match:
                params['amount'] = float(amount_match.group(1))

        elif tool_name == '计算器工具':
            # 改进数学表达式提取
            if '计算' in question or '等于' in question or '算式' in question:
                # 提取包含数学运算符的部分
                # 匹配包含数字和运算符的连续字符串
                math_patterns = [
                    r'(\d+(?:\s*[+\-*/^]\s*\d+)+)',  # 基础运算
                    r'(\d+\s*的\s*平方)',             # 平方
                    r'(\d+\s*的\s*立方)',             # 立方
                    r'(\d+\s*[+\-*/^]\s*\d+)'        # 简单运算
                ]
                
                for pattern in math_patterns:
                    match = re.search(pattern, question)
                    if match:
                        expr = match.group(1)
                        # 清理表达式
                        expr = expr.replace('的平方', '**2').replace('的立方', '**3')
                        expr = re.sub(r'\s+', '', expr)  # 移除空格
                        params['expression'] = expr
                        break
                
                # 如果没有匹配到特定模式，尝试提取数字和运算符
                if 'expression' not in params:
                    # 提取问题中的数学部分
                    calc_text = question
                    # 移除常见非数学词汇
                    for word in ['计算', '等于多少', '算式', '结果', '是多少']:
                        calc_text = calc_text.replace(word, '')
                    calc_text = calc_text.strip()
                    
                    # 如果只剩下数字和运算符，使用它
                    if re.match(r'^[\d+\-*/().^]+$', calc_text):
                        params['expression'] = calc_text

        return params

    def recognize_intent_by_rule(self, question: str) -> dict[str, any]:
        """识别用户意图 - 根据DeepSeek V3模型逻辑"""
        result = {}
        
        database_queries = []
        for table_name, keywords in self.database_keywords.items():
            if any(keyword in question for keyword in keywords):
                database_queries.append(table_name)
        
        if database_queries:
            # 如果包含数据库查询关键词，返回数据库查询格式
            return {'查询工具': '数据库'}
        
        # 2. 检查API工具
        recognized_tools = {}
        for tool_name, tool_info in self.api_tools_mapping.items():
            # 检查问题是否包含该工具的关键词
            if any(keyword in question for keyword in tool_info['keywords']):
                # 提取参数
                params = self.extract_parameters(question, tool_name)
                
                # 检查必填参数是否齐全
                missing_params = [p for p in tool_info['required_params'] if p not in params]
                if not missing_params:
                    recognized_tools[tool_name] = params
        
        # 3. 根据识别到的工具数量返回不同格式
        if len(recognized_tools) == 1:
            tool_name, params = list(recognized_tools.items())[0]
            return {tool_name: params}
        elif len(recognized_tools) > 1:
            return recognized_tools
        else:
            # 没有匹配到任何工具
            return {}

    def recognize_intent_by_api(self, question: str) -> dict[str, any]:
        """使用DeepSeek V3 API进行意图识别"""
        # DeepSeek API配置
        api_key = "sk-afabddf5ba0c4c96abb3567aa3324605"
        api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # 构建系统提示词
        system_prompt = {
            "任务说明": "你是一个意图识别专家，需要分析用户问题并判断应该查询数据库还是调用API工具",
            "处理规则": {
                "数据库查询": {
                    "触发条件": "当问题涉及交易记录、机构信息、商户信息时",
                    "返回格式": {
                        "数据库": "根据问题生成的MySQL查询SQL语句"
                    },
                    "表结构说明": {
                        "数据库表结构": [
                            {
                            "表名": "transaction_flow",
                            "注释": "交易流水表",
                            "字段": [
                                {"名称": "transaction_id", "类型": "VARCHAR(64)", "说明": "交易流水唯一标识"},
                                {"名称": "merchant_id", "类型": "VARCHAR(32)", "说明": "商户ID"},
                                {"名称": "institution_id", "类型": "VARCHAR(32)", "说明": "发起机构/支付机构ID"},
                                {"名称": "account_id", "类型": "VARCHAR(32)", "说明": "交易账户ID（付款方）"},
                                {"名称": "counterparty_id", "类型": "VARCHAR(32)", "说明": "对手方账户ID（收款方）"},
                                {"名称": "transaction_amount", "类型": "DECIMAL(15,2)", "说明": "交易金额（人民币）"},
                                {"名称": "transaction_type", "类型": "VARCHAR(20)", "说明": "交易类型（PAYMENT/REFUND）"},
                                {"名称": "transaction_time", "类型": "DATETIME", "说明": "交易发生时间"},
                                {"名称": "status", "类型": "VARCHAR(15)", "说明": "交易状态"},
                                {"名称": "remark", "类型": "VARCHAR(255)", "说明": "交易备注"}
                            ]
                            },
                            {
                            "表名": "institution_info",
                            "注释": "机构信息表",
                            "字段": [
                                {"名称": "institution_id", "类型": "VARCHAR(32)", "说明": "机构唯一标识"},
                                {"名称": "institution_name", "类型": "VARCHAR(100)", "说明": "机构名称"},
                                {"名称": "institution_type", "类型": "VARCHAR(20)", "说明": "机构类型（BANK/PAYMENT）"},
                                {"名称": "license_no", "类型": "VARCHAR(50)", "说明": "金融许可证号"},
                                {"名称": "legal_entity", "类型": "VARCHAR(100)", "说明": "法人代表"},
                                {"名称": "contact_phone", "类型": "VARCHAR(20)", "说明": "联系电话"},
                                {"名称": "contact_email", "类型": "VARCHAR(100)", "说明": "联系邮箱"},
                                {"名称": "status", "类型": "VARCHAR(15)", "说明": "机构状态"},
                                {"名称": "create_time", "类型": "DATETIME", "说明": "创建时间"}
                            ]
                            },
                            {
                            "表名": "merchant_info",
                            "注释": "商户信息表",
                            "字段": [
                                {"名称": "merchant_id", "类型": "VARCHAR(32)", "说明": "商户唯一标识"},
                                {"名称": "merchant_name", "类型": "VARCHAR(100)", "说明": "商户名称"},
                                {"名称": "merchant_type", "类型": "VARCHAR(20)", "说明": "商户类型（RETAIL/ONLINE）"},
                                {"名称": "merchant_category", "类型": "VARCHAR(10)", "说明": "商户类别码 MCC"},
                                {"名称": "legal_person", "类型": "VARCHAR(50)", "说明": "法人姓名"},
                                {"名称": "business_license", "类型": "VARCHAR(50)", "说明": "营业执照号"},
                                {"名称": "settlement_account", "类型": "VARCHAR(32)", "说明": "结算账户ID"},
                                {"名称": "status", "类型": "VARCHAR(15)", "说明": "商户状态"},
                                {"名称": "register_time", "类型": "DATETIME", "说明": "注册时间"}
                            ]
                            }
                        ],
                        "表关系": "transaction_flow.merchant_id → merchant_info | transaction_flow.institution_id → institution_info"
                        },
                "API工具调用": {
                    "触发条件": "当问题涉及以下特定功能时",
                    "工具列表": {
                        "信用卡服务": {
                            "功能": "查询信用卡月度账单",
                            "参数示例": {"cardNumber": "6211111111111111", "month": "2025-09"},
                            "关键词": ["信用卡", "账单", "月账单"]
                        },
                        "汇率服务": {
                            "功能": "汇率查询和货币转换",
                            "参数示例": {"fromCurrency": "USD", "toCurrency": "CNY", "amount": 100},
                            "关键词": ["汇率", "兑换", "货币", "等于多少"]
                        },
                        "水电煤服务": {
                            "功能": "查询水电煤月度账单",
                            "参数示例": {"householdId": "BJ001234567", "month": "2025-09", "utilityType": "electricity"},
                            "关键词": ["水电煤", "电费", "水费", "煤气费", "户号"]
                        },
                        "用户资产服务": {
                            "功能": "查询用户资产信息",
                            "参数示例": {"customerId": "110101199003072845", "assetType": "card"},
                            "关键词": ["用户资产", "资产信息"]
                        },
                        "支付订单服务": {
                            "功能": "创建支付订单",
                            "参数示例": {"merchantId": "M123456", "orderId": "ORD2025001", "amount": 100.50},
                            "关键词": ["支付订单", "创建订单"]
                        },
                        "获取当前日期工具": {
                            "功能": "获取当前日期",
                            "参数示例": {},
                            "关键词": ["当前日期", "现在时间", "今天日期"]
                        },
                        "计算器工具": {
                            "功能": "数学计算",
                            "参数示例": {"expression": "55**2+10"},
                            "关键词": ["计算", "等于多少", "算式"]
                        }
                    },
                    "返回格式": "单个工具: {工具名称: {参数键: 参数值}}，多个工具: {工具1: {参数}, 工具2: {参数}}"
                },
                "无匹配": {
                    "条件": "问题不匹配任何数据库或API工具",
                    "返回格式": {}
                }
            },
            "处理步骤": [
                "1. 分析用户问题的关键词和意图",
                "2. 判断是否涉及数据库查询（交易、机构、商户相关）",
                "3. 如不涉及数据库，检查是否匹配API工具",
                "4. 提取相应的参数信息",
                "5. 按指定格式返回JSON结果"
            ],
            "示例": {
                "数据库查询示例": {
                    "问题": "查询商户M123456在2025年8月的交易总额",
                    "返回": {"数据库": "SELECT SUM(transaction_amount) FROM transaction_flow WHERE merchant_id = 'M123456' AND transaction_time LIKE '2025-08%'"}
                },
                "API工具示例": {
                    "问题": "查询户号BJ001234568在2025-08的电使用量",
                    "返回": {"水电煤服务": {"householdId": "BJ001234568", "month": "2025-08", "utilityType": "electricity"}}
                },
                "多工具示例": {
                    "问题": "先查汇率再计算金额",
                    "返回": {"汇率服务": {"fromCurrency": "USD", "toCurrency": "CNY", "amount": 100}, "计算器工具": {"expression": "100 * 6.5"}}
                }
            },
            "输出要求": "只返回JSON格式结果，不要添加任何解释性文字"
        }
    }
        
        # 构建请求数据
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": json.dumps(system_prompt, ensure_ascii=False)
                },
                {
                    "role": "user", 
                    "content": question
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            # 发送API请求
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            # print(content)
            prefix = '```json'
            if content.startswith(prefix):
                content = content[len(prefix):-3]
            
            
            # 尝试解析JSON响应
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 如果响应不是有效的JSON，使用本地识别作为备选
                print(f"API响应不是有效JSON，使用本地识别: {content}")
                
                return self.recognize_intent_by_rule(question)
                
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API请求失败: {e}，使用本地识别")
            return self.recognize_intent_by_rule(question)
        except Exception as e:
            print(f"DeepSeek API处理错误: {e}，使用本地识别")
            return self.recognize_intent_by_rule(question)

    def recognize_intent(self, question: str) -> dict[str, any]:
        """意图识别主方法 - 默认使用API识别，失败时使用规则识别"""
        return self.recognize_intent_by_api(question)
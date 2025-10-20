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

    def recognize_intent_by_api(self, question: str, api_choice: str = "qwen") -> dict[str, any]:
        """使用AI API进行意图识别
        
        Args:
            question: 用户问题
            api_choice: API选择，支持 "deepseek" 或 "qwen"
        """
        if api_choice == "qwen":
            # 通义千问API配置 - 根据提供的curl配置修改
            api_key = "apikey_6ae236c214b79f62842bd42a207da07g"
            api_url = "http://122.144.165.92:8000/v1/chat/completions"
            model = "Qwen3-32B"
        else:
            # DeepSeek API配置
            api_key = "sk-afabddf5ba0c4c96abb3567aa3324605"
            api_url = "https://api.deepseek.com/v1/chat/completions"
            model = "deepseek-chat"
        
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
                                {"名称": "transaction_amount", "类型": "DECIMAL(15,2)", "说明": "交易金额，单位：元，币种为人民币"},
                                {"名称": "transaction_type", "类型": "VARCHAR(20)", "说明": "交易类型，如 PAYMENT, REFUND"},
                                {"名称": "transaction_time", "类型": "DATETIME", "说明": "交易发生时间"},
                                {"名称": "status", "类型": "VARCHAR(15)", "说明": "交易状态，如 SUCCESS, FAILED"},
                                {"名称": "remark", "类型": "VARCHAR(255)", "说明": "交易备注"}
                            ],
                            "数据案例": {
                                "transaction_id": "T00000001",
                                "merchant_id": "M00006",
                                "institution_id": "INST0001",
                                "account_id": "ACC000179",
                                "counterparty_id": "ACC000544",
                                "transaction_amount": 42537.71,
                                "transaction_type": "WITHDRAW",
                                "transaction_time": "2021-02-01 12:04:01",
                                "status": "PENDING",
                                "remark": "交易备注0001"
                            }
                            },
                            {
                            "表名": "institution_info",
                            "注释": "机构信息表",
                            "字段": [
                                {"名称": "institution_id", "类型": "VARCHAR(32)", "说明": "机构唯一标识"},
                                {"名称": "institution_name", "类型": "VARCHAR(100)", "说明": "机构名称"},
                                {"名称": "institution_type", "类型": "VARCHAR(20)", "说明": "机构类型，如 BANK, PAYMENT"},
                                {"名称": "license_no", "类型": "VARCHAR(50)", "说明": "金融许可证号"},
                                {"名称": "legal_entity", "类型": "VARCHAR(100)", "说明": "法人代表"},
                                {"名称": "contact_phone", "类型": "VARCHAR(20)", "说明": "联系电话"},
                                {"名称": "contact_email", "类型": "VARCHAR(100)", "说明": "联系邮箱"},
                                {"名称": "status", "类型": "VARCHAR(15)", "说明": "机构状态，如 ACTIVE"},
                                {"名称": "create_time", "类型": "DATETIME", "说明": "创建时间"}
                            ],
                            "数据案例": {
                                "institution_id": "INST0001",
                                "institution_name": "机构01",
                                "institution_type": "BANK",
                                "license_no": "LIC000001",
                                "legal_entity": "法人01",
                                "contact_phone": "13800013667",
                                "contact_email": "contact01@inst.com",
                                "status": "ACTIVE",
                                "create_time": "2022-10-01 17:57:59"
                            }
                            },
                            {
                            "表名": "merchant_info",
                            "注释": "商户信息表",
                            "字段": [
                                {"名称": "merchant_id", "类型": "VARCHAR(32)", "说明": "商户唯一标识"},
                                {"名称": "merchant_name", "类型": "VARCHAR(100)", "说明": "商户名称"},
                                {"名称": "merchant_type", "类型": "VARCHAR(20)", "说明": "商户类型，如 RETAIL, ONLINE"},
                                {"名称": "merchant_category", "类型": "VARCHAR(10)", "说明": "商户类别码 MCC"},
                                {"名称": "legal_person", "类型": "VARCHAR(50)", "说明": "法人姓名"},
                                {"名称": "business_license", "类型": "VARCHAR(50)", "说明": "营业执照号"},
                                {"名称": "settlement_account", "类型": "VARCHAR(32)", "说明": "结算账户ID"},
                                {"名称": "status", "类型": "VARCHAR(15)", "说明": "商户状态，如 ACTIVE"},
                                {"名称": "register_time", "类型": "DATETIME", "说明": "注册时间"}
                            ],
                            "数据案例": {
                                "merchant_id": "M00001",
                                "merchant_name": "商户01",
                                "merchant_type": "SERVICE",
                                "merchant_category": "5964",
                                "legal_person": "法人01",
                                "business_license": "BL00000001",
                                "settlement_account": "ACC000157",
                                "status": "INACTIVE",
                                "register_time": "2022-08-29 15:15:46"
                            }
                            },
                            {
                            "表名": "account_info",
                            "注释": "账户信息表",
                            "字段": [
                                {"名称": "account_id", "类型": "VARCHAR(32)", "说明": "账户唯一标识"},
                                {"名称": "account_name", "类型": "VARCHAR(100)", "说明": "账户名称"},
                                {"名称": "account_type", "类型": "VARCHAR(20)", "说明": "账户类型，如 SETTLEMENT"},
                                {"名称": "bank_name", "类型": "VARCHAR(100)", "说明": "开户银行"},
                                {"名称": "bank_account_no", "类型": "VARCHAR(34)", "说明": "银行账号"},
                                {"名称": "status", "类型": "VARCHAR(15)", "说明": "账户状态，如 ACTIVE"},
                                {"名称": "create_time", "类型": "DATETIME", "说明": "账户开立时间"}
                            ],
                            "数据案例": {
                                "account_id": "ACC000001",
                                "account_name": "账户001",
                                "account_type": "SAVING",
                                "bank_name": "机构17",
                                "bank_account_no": "622202884301003517",
                                "status": "FROZEN",
                                "create_time": "2023-07-19 16:53:45"
                            }
                            }
                        ],
                        "表关系": "transaction_flow.merchant_id → merchant_info | transaction_flow.institution_id → institution_info | transaction_flow.account_id → account_info | merchant_info.settlement_account → account_info"
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
                "5. 针对于多工具的情况，需要判断是否涉及前后流程依赖关系，如果涉及参考示例里面的工具依赖调用示例", 
                "6. 按指定格式返回JSON结果"
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
                },
                "工具依赖调用示例": {
                    "问题": "请查询交易类型为PAYMENT且交易金额大于500元的最近交易金额，以此为订单金额，为商户M222222创建订单号为ORD2025005的支付订单，返回支付订单ID。",
                    "返回": {
                        "工具依赖调用": {
                            "steps": [
                                {"数据库": "SELECT transaction_amount FROM transaction_flow WHERE transaction_type = 'PAYMENT' AND transaction_amount > 500 ORDER BY transaction_time DESC LIMIT 1"},
                                {"支付订单服务": {"merchantId": "M222222", "orderId": "ORD2025005", "param_mapping": {"amount": "step1.result.transaction_amount"}}}
                            ]
                        }
                    }
                }
            },
            "输出要求": "只返回JSON格式结果，不要添加任何解释性文字"
            }
        }
        system_prompt = """你是一个智能任务路由专家，需要根据用户的问题判断应执行数据库查询还是调用某个API工具。请严格按照以下规则进行分析和输出。
            一、判断原则
            优先判断是否涉及数据库查询
            如果问题中提到“交易”、“商户”、“机构”、“账户”、“流水”等相关信息，请生成对应的 MySQL 查询语句。

            否则判断是否匹配某个API工具
            查看问题是否涉及账单查询、汇率转换、支付创建等特定功能，若匹配，则提取参数并调用对应工具。

            如果都不匹配，返回空对象

            二、数据库查询规则
            当问题涉及交易记录、商户信息、机构信息或账户信息时，使用以下表结构生成 SQL 查询语句。

            表结构说明：
            transaction_flow（交易流水表）

            字段：transaction_id, merchant_id, institution_id, account_id, counterparty_id, transaction_amount(金额), transaction_type(类型: PAYMENT/REFUND/WITHDRAW), transaction_time, status, remark
            merchant_info（商户信息表）

            字段：merchant_id, merchant_name, merchant_type, merchant_category(MCC码), legal_person, business_license, settlement_account, status, register_time
            institution_info（机构信息表）

            字段：institution_id, institution_name, institution_type(BANK/PAYMENT), license_no, legal_entity, contact_phone, contact_email, status, create_time
            account_info（账户信息表）

            字段：account_id, account_name, account_type, bank_name, bank_account_no, status, create_time
            关联关系：
            transaction_flow.merchant_id → merchant_info.merchant_id
            transaction_flow.institution_id → institution_info.institution_id
            transaction_flow.account_id → account_info.account_id
            merchant_info.settlement_account → account_info.account_id
            输出格式为：{"数据库": "SELECT ..."}

            三、API工具调用规则
            如果问题不涉及上述数据库内容，但包含以下关键词，请调用相应工具。

            工具名称	触发关键词	功能	参数示例
            信用卡服务	信用卡、账单、月账单	查询信用卡账单	{"cardNumber": "6211111111111111", "month": "2025-09"}
            汇率服务	汇率、兑换、等于多少、货币	货币兑换计算	{"fromCurrency": "USD", "toCurrency": "CNY", "amount": 100}
            水电煤服务	水电煤、电费、水费、煤气费、户号	查询公用事业账单	{"householdId": "BJ001234567", "month": "2025-08", "utilityType": "electricity"}
            用户资产服务	用户资产、资产信息	查询用户名下资产	{"customerId": "身份证号", "assetType": "card"}
            支付订单服务	支付订单、创建订单	创建一个支付订单	{"merchantId": "M123456", "orderId": "ORD2025001", "amount": 100.50}
            获取当前日期工具	当前日期、现在时间、今天日期	返回当前系统日期	{}
            计算器工具	计算、算式、等于多少	执行数学表达式	{"expression": "55**2+10"}
            多个工具同时调用时，按顺序列出：
            {"工具A": {参数}, "工具B": {参数}}

            若存在依赖关系（如先查数据再用于下一步），请按步骤组织为：
            {
            "工具依赖调用": {
                "steps": [
                {
                    "数据库": "SELECT ..."
                },
                {
                    "支付订单服务": {
                    "param_mapping": {
                        "amount": "step1.result.xxx"
                    }
                    }
                }
                ]
            }
            }
            四、处理流程
            分析用户问题中的关键词与意图。
            判断是否涉及交易、商户、机构等信息 → 是 → 生成SQL。
            否则检查是否匹配任一API工具关键词。
            提取所需参数，注意日期、编号、金额等关键值。
            若多个工具需协同执行，判断是否有前后依赖；如有，使用“工具依赖调用”格式。
            最终只输出 JSON 格式结果，不要任何解释文字。
            五、示例参考
            示例1：数据库查询
            问题：查询商户M123456在2025年8月的交易总额
            返回：{"数据库": "SELECT SUM(transaction_amount) FROM transaction_flow WHERE merchant_id = 'M123456' AND transaction_time LIKE '2025-08%'"}

            示例2：API工具调用
            问题：查询户号BJ001234568在2025-08的电使用量
            返回：{"水电煤服务": {"householdId": "BJ001234568", "month": "2025-08", "utilityType": "electricity"}}

            示例3：多工具并行
            问题：先查汇率再计算金额
            返回：{"汇率服务": {"fromCurrency": "USD", "toCurrency": "CNY", "amount": 100}, "计算器工具": {"expression": "100 * 6.5"}}

            示例4：工具依赖调用
            问题：请查询交易类型为PAYMENT且交易金额大于500元的最近交易金额，以此为订单金额，为商户M222222创建订单号为ORD2025005的支付订单
            返回： { "工具依赖调用": { "steps": [ {"数据库": "SELECT transaction_amount FROM transaction_flow WHERE transaction_type = 'PAYMENT' AND transaction_amount > 500 ORDER BY transaction_time DESC LIMIT 1"}, {"支付订单服务": {"merchantId": "M222222", "orderId": "ORD2025005", "param_mapping": {"amount": "step1.result.transaction_amount"}}} ] } }

            六、最终输出要求
            只返回合法的 JSON 对象
            不要添加任何说明、注释或额外文本
            确保字段名和格式完全正确
            """
        
        system_prompt = """你是一个智能任务路由专家，需要根据用户的问题判断应执行数据库查询还是调用某个API工具。请严格按照以下规则进行分析和输出。
            一、判断原则
            优先判断是否涉及数据库查询
            如果问题中提到“交易”、“商户”、“机构”、“账户”、“流水”等相关信息，请生成对应的 MySQL 查询语句。

            否则判断是否匹配某个API工具
            查看问题是否涉及账单查询、汇率转换、支付创建等特定功能，若匹配，则提取参数并调用对应工具。

            如果都不匹配，返回空对象

            二、数据库查询规则
            当问题涉及交易记录、商户信息、机构信息或账户信息时，使用以下表结构生成 SQL 查询语句。

            表结构说明：
            CREATE TABLE wide_table (
            transaction_id VARCHAR(64) PRIMARY KEY COMMENT '交易流水唯一标识',
            merchant_id VARCHAR(32) NOT NULL COMMENT '商户ID',
            institution_id VARCHAR(32) NOT NULL COMMENT '机构ID',
            account_id VARCHAR(32) NOT NULL COMMENT '交易账户ID',
            counterparty_id VARCHAR(32) NOT NULL COMMENT '对手方账户ID',
            transaction_amount DECIMAL(15,2) NOT NULL COMMENT '交易金额，单位：元',
            transaction_type VARCHAR(20) NOT NULL COMMENT '交易类型',
            transaction_time DATETIME NOT NULL COMMENT '交易发生时间',
            status_ VARCHAR(15) NOT NULL COMMENT '交易状态',
            remark VARCHAR(255) COMMENT '交易备注',

            merchant_name VARCHAR(100) COMMENT '商户名称',
            merchant_type VARCHAR(20) COMMENT '商户类型',
            merchant_category VARCHAR(10) COMMENT '商户类别码MCC',
            legal_person VARCHAR(50) COMMENT '法人姓名',
            business_license VARCHAR(50) COMMENT '营业执照号',
            merchant_status VARCHAR(15) COMMENT '商户状态',
            register_time DATETIME COMMENT '注册时间',

            institution_name VARCHAR(100) COMMENT '机构名称',
            institution_type VARCHAR(20) COMMENT '机构类型',
            license_no VARCHAR(50) COMMENT '金融许可证号',
            legal_entity VARCHAR(100) COMMENT '法人代表',
            contact_phone VARCHAR(20) COMMENT '联系电话',
            contact_email VARCHAR(100) COMMENT '联系邮箱',
            institution_status VARCHAR(15) COMMENT '机构状态',
            create_time DATETIME COMMENT '创建时间',

            account_name VARCHAR(100) COMMENT '账户名称',
            account_type VARCHAR(20) COMMENT '账户类型',
            bank_name VARCHAR(100) COMMENT '开户银行',
            bank_account_no VARCHAR(34) COMMENT '银行账号',
            account_status VARCHAR(15) COMMENT '账户状态',
            account_create_time DATETIME COMMENT '账户开立时间',

            settlement_account VARCHAR(32) COMMENT '结算账户ID',
            counterparty_account_name VARCHAR(100) COMMENT '对手方账户名称',
            counterparty_bank_name VARCHAR(100) COMMENT '对手方开户银行',

            INDEX idx_merchant_id (merchant_id),
            INDEX idx_institution_id (institution_id),
            INDEX idx_account_id (account_id),
            INDEX idx_transaction_time (transaction_time),
            INDEX idx_transaction_type (transaction_type),
            INDEX idx_status (status_)
            ) COMMENT '大宽表：整合交易流水、商户信息、机构信息、账户信息的宽表'
            ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT '大宽表：整合交易流水、商户信息、机构信息、账户信息的宽表';

            输出格式为：{"数据库": "SELECT ..."}

            三、API工具调用规则
            如果问题不涉及上述数据库内容，但包含以下关键词，请调用相应工具。

            工具名称	触发关键词	功能	参数示例
            信用卡服务	信用卡、账单、月账单	查询信用卡账单	{"cardNumber": "6211111111111111", "month": "2025-09"}
            汇率服务	汇率、兑换、等于多少、货币	货币兑换计算	{"fromCurrency": "USD", "toCurrency": "CNY", "amount": 100}
            水电煤服务	水电煤、电费、水费、煤气费、户号	查询公用事业账单	{"householdId": "BJ001234567", "month": "2025-08", "utilityType": "electricity"}
            用户资产服务	用户资产、资产信息	查询用户名下资产	{"customerId": "身份证号", "assetType": "card"}
            支付订单服务	支付订单、创建订单	创建一个支付订单	{"merchantId": "M123456", "orderId": "ORD2025001", "amount": 100.50}
            获取当前日期工具	当前日期、现在时间、今天日期	返回当前系统日期	{}
            计算器工具	计算、算式、等于多少	执行数学表达式	{"expression": "55**2+10"}
            多个工具同时调用时，按顺序列出：
            {"工具A": {参数}, "工具B": {参数}}

            若存在依赖关系（如先查数据再用于下一步），请按步骤组织为：
            {
            "工具依赖调用": {
                "steps": [
                {
                    "数据库": "SELECT ..."
                },
                {
                    "支付订单服务": {
                    "param_mapping": {
                        "amount": "step1.result.xxx"
                    }
                    }
                }
                ]
            }
            }
            四、处理流程
            分析用户问题中的关键词与意图。
            判断是否涉及交易、商户、机构等信息 → 是 → 生成SQL。
            否则检查是否匹配任一API工具关键词。
            提取所需参数，注意日期、编号、金额等关键值。
            若多个工具需协同执行，判断是否有前后依赖；如有，使用“工具依赖调用”格式。
            最终只输出 JSON 格式结果，不要任何解释文字。
            五、示例参考
            示例1：数据库查询
            问题：查询商户M123456在2025年8月的交易总额
            返回：{"数据库": "SELECT SUM(transaction_amount) FROM transaction_flow WHERE merchant_id = 'M123456' AND transaction_time LIKE '2025-08%'"}

            示例2：API工具调用
            问题：查询户号BJ001234568在2025-08的电使用量
            返回：{"水电煤服务": {"householdId": "BJ001234568", "month": "2025-08", "utilityType": "electricity"}}

            示例3：多工具并行
            问题：先查汇率再计算金额
            返回：{"汇率服务": {"fromCurrency": "USD", "toCurrency": "CNY", "amount": 100}, "计算器工具": {"expression": "100 * 6.5"}}

            示例4：工具依赖调用
            问题：请查询交易类型为PAYMENT且交易金额大于500元的最近交易金额，以此为订单金额，为商户M222222创建订单号为ORD2025005的支付订单
            返回： { "工具依赖调用": { "steps": [ {"数据库": "SELECT transaction_amount FROM transaction_flow WHERE transaction_type = 'PAYMENT' AND transaction_amount > 500 ORDER BY transaction_time DESC LIMIT 1"}, {"支付订单服务": {"merchantId": "M222222", "orderId": "ORD2025005", "param_mapping": {"amount": "step1.result.transaction_amount"}}} ] } }

            六、最终输出要求
            只返回合法的 JSON 对象
            不要添加任何说明、注释或额外文本
            确保字段名和格式完全正确
            """
        

        # 构建请求数据
        if api_choice == "qwen":
            # 通义千问API请求格式 - 使用API密钥认证
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
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
                "max_tokens": 512,
                "presence_penalty": 1.03,
                "frequency_penalty": 1.0,
                "seed": None,
                "temperature": 0.5,
                "top_p": 0.95,
                "stream": False
            }
        else:
            # DeepSeek API请求格式
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
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
            
            if api_choice == "qwen":
                # 通义千问API响应格式 - 使用API密钥方式的响应结构
                content = result["choices"][0]["message"]["content"]
            else:
                # DeepSeek API响应格式
                content = result["choices"][0]["message"]["content"]
            
            # 清理JSON格式
            print('===返回的 content ===')
            print(content)
            print('===返回的 content 结束 ===')
            prefix = '```json'
            if content.startswith(prefix):
                content = content[len(prefix):-3]
            
            # 尝试解析JSON响应
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试从```json```代码块中提取内容
                try:
                    # 查找```json开头和```结尾的内容
                    json_pattern = r'```json\s*(.*?)\s*```'
                    matches = re.findall(json_pattern, content, re.DOTALL)
                    
                    if matches:
                        # 使用第一个匹配的JSON内容
                        json_content = matches[0].strip()
                        return json.loads(json_content)
                    else:
                        # 如果没有找到代码块，尝试直接解析整个内容
                        raise json.JSONDecodeError("No JSON code block found", content, 0)
                        
                except json.JSONDecodeError:
                    # 如果从代码块解析也失败，使用本地识别作为备选
                    print(f"API响应不是有效JSON，使用本地识别: {content}")
                    return self.recognize_intent_by_rule(question)
                
        except requests.exceptions.RequestException as e:
            print(f"{api_choice.upper()} API请求失败: {e}，使用本地识别")
            return self.recognize_intent_by_rule(question)
        except Exception as e:
            print(f"{api_choice.upper()} API处理错误: {e}，使用本地识别")
            return self.recognize_intent_by_rule(question)

    def recognize_intent(self, question: str, api_choice: str = "deepseek") -> dict[str, any]:
        """意图识别主方法 - 默认使用API识别，失败时使用规则识别
        
        Args:
            question: 用户问题
            api_choice: API选择，支持 "deepseek" 或 "qwen"
        """
        return self.recognize_intent_by_api(question, api_choice)

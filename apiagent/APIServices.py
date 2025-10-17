import requests
import json
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal
import re

# 测试模式配置 - 当网络连接不通时设置为 True
TEST_MODE = True

class APIToolClient:
    """API工具客户端基类"""

    def __init__(self, base_url: str, app_id: str, app_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'X-App-Id': app_id,
            'X-App-Key': app_key,
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """统一的GET请求方法"""
        # 测试模式下返回模拟数据
        if TEST_MODE:
            return self._get_mock_data(endpoint, params)
        
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'error': f'API请求失败: {str(e)}'}

    def _get_mock_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """测试模式下的模拟数据返回"""
        # 默认模拟数据，子类可以重写此方法提供特定模拟数据
        return {
            'status': 'success',
            'message': '测试模式 - 模拟数据',
            'endpoint': endpoint,
            'params': params,
            'data': {}
        }

    def _validate_parameters(self, params: Dict[str, Any], required_fields: List[str]) -> bool:
        """验证必填参数"""
        for field in required_fields:
            if field not in params or params[field] is None:
                return False
        return True


class CreditCardService(APIToolClient):
    """信用卡服务"""

    def get_monthly_bill(self, cardNumber: str, month: str) -> Dict[str, Any]:
        """查询月度账单信息"""
        endpoint = "/api/credit-card/monthly-bill"
        params = {
            'cardNumber': cardNumber,
            'month': month
        }

        if not self._validate_parameters(params, ['cardNumber', 'month']):
            return {'error': '缺少必填参数: cardNumber 或 month'}

        return self._get(endpoint, params)

    def _get_mock_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """信用卡服务模拟数据"""
        return {
            'status': 'success',
            'cardNumber': params.get('cardNumber', ''),
            'month': params.get('month', ''),
            'totalAmount': 1250.50,
            'currency': 'CNY',
            'transactions': [
                {'date': f"{params.get('month', '')}-01", 'amount': 150.00, 'merchant': '超市购物'},
                {'date': f"{params.get('month', '')}-05", 'amount': 300.00, 'merchant': '餐厅消费'},
                {'date': f"{params.get('month', '')}-15", 'amount': 800.50, 'merchant': '在线购物'}
            ],
            'dueDate': f"{params.get('month', '')}-25",
            'minPayment': 125.05
        }


class ExchangeRateService(APIToolClient):
    """汇率服务"""

    SUPPORTED_CURRENCIES = {'USD', 'CNY', 'EUR', 'JPY', 'GBP', 'KRW'}

    def parse_currency_query(self, query: str) -> dict:
        """解析货币兑换查询问题"""
        currency_mapping = {
            '日元': 'JPY', '円': 'JPY', '日圆': 'JPY',
            '韩元': 'KRW', '韩圜圜': 'KRW', '韩圆': 'KRW', '韩币': 'KRW',
            '人民币': 'CNY', '元': 'CNY', '人民币元': 'CNY',
            '美元': 'USD', '美金': 'USD', '美圆': 'USD',
            '欧元': 'EUR', '欧圆': 'EUR',
            '英镑': 'GBP', '英磅': 'GBP'
        }

        patterns = [
            r'(\d+(?:\.\d+)?)\s*([^\d\s]+)\s*等于\s*多少\s*([^\d\s]+)',
            r'(\d+(?:\.\d+)?)\s*([^\d\s]+)\s*兑换\s*([^\d\s]+)',
            r'(\d+(?:\.\d+)?)\s*([^\d\s]+)\s*换成\s*([^\d\s]+)',
            r'(\d+(?:\.\d+)?)\s*([^\d\s]+)\s*可以换多少\s*([^\d\s]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                amount = float(match.group(1))
                from_currency_name = match.group(2).strip()
                to_currency_name = match.group(3).strip()

                from_currency = currency_mapping.get(from_currency_name)
                to_currency = currency_mapping.get(to_currency_name)

                if from_currency and to_currency:
                    return {
                        'fromCurrency': from_currency,
                        'toCurrency': to_currency,
                        'amount': amount
                    }
                else:
                    return {
                        'fromCurrency': from_currency_name,
                        'toCurrency': to_currency_name,
                        'amount': amount,
                        'error': '无法识别的货币名称，请使用标准货币代码'
                    }

        return {'error': '无法解析查询格式'}

    def query_to_exchange_rate_params(self, query: str) -> dict:
        """将自然语言查询转换为汇率查询参数"""
        parsed_params = self.parse_currency_query(query)

        if 'error' in parsed_params:
            return parsed_params

        if parsed_params['fromCurrency'] not in self.SUPPORTED_CURRENCIES:
            return {'error': f'不支持的源货币: {parsed_params["fromCurrency"]}'}

        if parsed_params['toCurrency'] not in self.SUPPORTED_CURRENCIES:
            return {'error': f'不支持的目标货币: {parsed_params["toCurrency"]}'}

        return parsed_params

    def get_exchange_rate(self, fromCurrency: str, toCurrency: str, amount: float = 1.0) -> Dict[str, Any]:
        """查询汇率和货币转换"""
        endpoint = "/api/exchange-rate"
        params = {
            'fromCurrency': fromCurrency.upper(),
            'toCurrency': toCurrency.upper(),
            'amount': amount
        }

        if params['fromCurrency'] not in self.SUPPORTED_CURRENCIES:
            return {'error': f'不支持的源货币代码: {fromCurrency}'}
        if params['toCurrency'] not in self.SUPPORTED_CURRENCIES:
            return {'error': f'不支持的目标货币代码: {toCurrency}'}

        if not self._validate_parameters(params, ['fromCurrency', 'toCurrency']):
            return {'error': '缺少必填参数: fromCurrency 或 toCurrency'}

        return self._get(endpoint, params)

    def _get_mock_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """汇率服务模拟数据"""
        # 汇率映射表
        exchange_rates = {
            ('USD', 'CNY'): 6.5,
            ('CNY', 'USD'): 0.15,
            ('JPY', 'CNY'): 0.045,
            ('CNY', 'JPY'): 22.22,
            ('EUR', 'CNY'): 7.2,
            ('CNY', 'EUR'): 0.14,
            ('GBP', 'CNY'): 8.1,
            ('CNY', 'GBP'): 0.12,
            ('KRW', 'CNY'): 0.005,
            ('CNY', 'KRW'): 200
        }
        
        from_currency = params.get('fromCurrency', '').upper()
        to_currency = params.get('toCurrency', '').upper()
        amount = params.get('amount', 1.0)
        
        rate = exchange_rates.get((from_currency, to_currency), 1.0)
        converted_amount = amount * rate
        
        return {
            'status': 'success',
            'fromCurrency': from_currency,
            'toCurrency': to_currency,
            'amount': amount,
            'exchangeRate': rate,
            'convertedAmount': round(converted_amount, 2),
            'timestamp': datetime.now().isoformat()
        }


class UtilityBillService(APIToolClient):
    """水电煤服务"""

    utilityTypeS = {'electricity', 'water', 'gas'}

    def get_monthly_bill(self, householdId: str, month: str, utilityType: str = 'electricity') -> Dict[str, Any]:
        """查询水电煤月度账单"""
        endpoint = "/api/utility-bill/monthly-bill"
        params = {
            'householdId': householdId,
            'month': month,
            'utilityType': utilityType
        }

        if not self._validate_parameters(params, ['householdId', 'month']):
            return {'error': '缺少必填参数: householdId 或 month'}

        if utilityType not in self.utilityTypeS:
            return {'error': f'不支持的账单类型: {utilityType}'}

        return self._get(endpoint, params)

    def _get_mock_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """水电煤服务模拟数据"""
        utility_type = params.get('utilityType', 'electricity')
        household_id = params.get('householdId', '')
        month = params.get('month', '')
        
        # 根据不同类型生成不同的模拟数据
        if utility_type == 'electricity':
            return {
                'status': 'success',
                'householdId': household_id,
                'month': month,
                'utilityType': 'electricity',
                'usage': 256.5,  # 度
                'amount': 128.25,  # 元
                'unitPrice': 0.5,  # 元/度
                'dueDate': f"{month}-20"
            }
        elif utility_type == 'water':
            return {
                'status': 'success',
                'householdId': household_id,
                'month': month,
                'utilityType': 'water',
                'usage': 15.8,  # 吨
                'amount': 47.4,  # 元
                'unitPrice': 3.0,  # 元/吨
                'dueDate': f"{month}-15"
            }
        else:  # gas
            return {
                'status': 'success',
                'householdId': household_id,
                'month': month,
                'utilityType': 'gas',
                'usage': 32.2,  # 立方米
                'amount': 96.6,  # 元
                'unitPrice': 3.0,  # 元/立方米
                'dueDate': f"{month}-25"
            }


class UserAssetsService(APIToolClient):
    """用户资产服务"""

    assetTypeS = {'card', 'household'}

    def get_user_assets(self, customerId: str, assetType: str = 'card') -> Dict[str, Any]:
        """查询用户资产信息"""
        endpoint = "/api/user/assets"
        params = {
            'customerId': customerId,
            'assetType': assetType
        }

        if not self._validate_parameters(params, ['customerId']):
            return {'error': '缺少必填参数: customerId'}

        if assetType not in self.assetTypeS:
            return {'error': f'不支持的资产类型: {assetType}'}

        return self._get(endpoint, params)

    def _get_mock_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """用户资产服务模拟数据"""
        customer_id = params.get('customerId', '')
        asset_type = params.get('assetType', 'card')
        
        if asset_type == 'card':
            return {
                'status': 'success',
                'customerId': customer_id,
                'assetType': 'card',
                'totalAssets': 125000.50,
                'cards': [
                    {
                        'cardNumber': '6211111111111111',
                        'cardType': '信用卡',
                        'bank': '中国银行',
                        'creditLimit': 50000.00,
                        'availableCredit': 37500.00,
                        'currentBalance': 12500.00
                    },
                    {
                        'cardNumber': '6222222222222222',
                        'cardType': '借记卡',
                        'bank': '工商银行',
                        'balance': 112500.50
                    }
                ]
            }
        else:  # household
            return {
                'status': 'success',
                'customerId': customer_id,
                'assetType': 'household',
                'totalAssets': 3500000.00,
                'properties': [
                    {
                        'address': '北京市朝阳区某某小区1号楼101',
                        'area': 85.6,
                        'estimatedValue': 2500000.00,
                        'purchaseDate': '2018-05-15'
                    },
                    {
                        'address': '上海市浦东新区某某路200号',
                        'area': 120.0,
                        'estimatedValue': 1000000.00,
                        'purchaseDate': '2020-12-20'
                    }
                ]
            }


class PaymentOrderService(APIToolClient):
    """支付订单服务"""

    def create_payment_order(self, merchantId: str, orderId: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """创建支付订单"""
        endpoint = "/api/qr/create-payment-order"
        params = {
            'merchantId': merchantId,
            'orderId': orderId
        }

        if amount is not None:
            params['amount'] = amount

        if not self._validate_parameters(params, ['merchantId', 'orderId']):
            return {'error': '缺少必填参数: merchantId 或 orderId'}

        return self._get(endpoint, params)

    def _get_mock_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """支付订单服务模拟数据"""
        merchant_id = params.get('merchantId', '')
        order_id = params.get('orderId', '')
        amount = params.get('amount', 0.0)
        
        return {
            'status': 'success',
            'merchantId': merchant_id,
            'orderId': order_id,
            'amount': amount,
            'orderStatus': 'CREATED',
            'paymentUrl': f'https://payment.example.com/pay/{order_id}',
            'qrCode': f'data:image/png;base64,模拟二维码数据',
            'createTime': datetime.now().isoformat(),
            'expireTime': (datetime.now() + timedelta(hours=24)).isoformat()
        }


class LocalTools:
    """本地工具类"""

    @staticmethod
    def get_current_date() -> Dict[str, Any]:
        """获取当前日期（东八区）"""
        beijing_tz = timezone(timedelta(hours=8))
        current_time = datetime.now(beijing_tz)

        return {
            'current_date': current_time.strftime('%Y-%m-%d'),
            'current_datetime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'timezone': 'UTC+8',
            'timestamp': int(current_time.timestamp())
        }

    @staticmethod
    def calculator(expression: str) -> Dict[str, Any]:
        """计算器工具"""
        try:
            replace_dict = {
                ' ': '', '^': '**', '的平方': '**2',
                '的立方': '**3', '是多少': '', '等于多少': ''
            }
            for key, value in replace_dict.items():
                expression = expression.replace(key, value)

            if not re.match(r'^[0-9+\-*/().,sqrtpowlogsincostan]+$', expression):
                return {'expression': expression, 'error': '表达式包含不安全字符'}

            expression = expression.replace('sqrt', 'math.sqrt')
            expression = expression.replace('pow', 'math.pow')
            expression = expression.replace('sin', 'math.sin')
            expression = expression.replace('cos', 'math.cos')
            expression = expression.replace('tan', 'math.tan')
            expression = expression.replace('log', 'math.log')

            result = eval(expression, {'math': math, '__builtins__': {}})

            return {
                'expression': expression,
                'result': result,
                'type': type(result).__name__
            }

        except ZeroDivisionError:
            return {'error': '除零错误'}
        except Exception as e:
            return {'error': f'计算错误: {str(e)}'}


class ToolService:
    """工具服务主类"""

    def __init__(self, base_url: str, app_id: str, app_key: str):
        self.credit_card = CreditCardService(base_url, app_id, app_key)
        self.exchange_rate = ExchangeRateService(base_url, app_id, app_key)
        self.utility_bill = UtilityBillService(base_url, app_id, app_key)
        self.user_assets = UserAssetsService(base_url, app_id, app_key)
        self.payment_order = PaymentOrderService(base_url, app_id, app_key)
        self.local_tools = LocalTools()
'''
Main script to fit model
and generate predictions

Andy Wheeler
'''

from src import feat, mod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

def create_features(df):
    """创建新特征"""
    # 卫星数据特征
    for scale in ['500', '1000', '2500']:
        if all(f'{c}_{scale}' in df.columns for c in ['r', 'g', 'b']):
            # RGB比率特征
            df[f'rgb_ratio_{scale}'] = df[f'r_{scale}'] / (df[f'g_{scale}'] + df[f'b_{scale}'] + 1e-6)
            df[f'rg_ratio_{scale}'] = df[f'r_{scale}'] / (df[f'g_{scale}'] + 1e-6)
            df[f'gb_ratio_{scale}'] = df[f'g_{scale}'] / (df[f'b_{scale}'] + 1e-6)
            
            # RGB差异特征
            df[f'rg_diff_{scale}'] = df[f'r_{scale}'] - df[f'g_{scale}']
            df[f'gb_diff_{scale}'] = df[f'g_{scale}'] - df[f'b_{scale}']
            
            # 强度特征
            df[f'intensity_{scale}'] = (df[f'r_{scale}'] + df[f'g_{scale}'] + df[f'b_{scale}']) / 3
    
    # 地理位置特征
    if all(col in df.columns for col in ['latitude', 'longitude', 'elevation', 'maxe']):
        df['lat_lon_ratio'] = df['latitude'] / (df['longitude'] + 1e-6)
        df['elevation_ratio'] = df['elevation'] / (df['maxe'] + 1e-6)
        df['elevation_diff'] = df['maxe'] - df['elevation']
    
    # 时间特征
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    return df

def analyze_features(data):
    """分析特征分布和相关性"""
    print("\n=== 特征分析 ===")
    
    # 1. 基本统计信息
    print("\n1. 特征基本统计信息：")
    print(data.describe())
    
    # 2. 缺失值分析
    print("\n2. 缺失值分析：")
    missing = data.isnull().sum()
    print(missing[missing > 0])
    
    # 3. 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data['severity'], bins=30)
    plt.title('目标变量分布')
    plt.savefig('severity_distribution.png')
    plt.close()
    
    # 4. 特征相关性分析
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def preprocess_data(data):
    """数据预处理"""
    print("\n=== 数据预处理 ===")
    
    # 1. 处理缺失值
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # 2. 标准化数值特征
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R2分数: {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def compare_models(models_dict, y_true):
    """比较不同模型的性能"""
    print("\n=== 模型性能比较 ===")
    print("模型\t\tMSE\t\tRMSE\t\tMAE\t\tR2")
    print("-" * 60)
    
    best_model = None
    best_rmse = float('inf')
    
    for name, pred in models_dict.items():
        metrics = evaluate_model(y_true, pred)
        print(f"{name:<12} {metrics['mse']:.4f}\t{metrics['rmse']:.4f}\t{metrics['mae']:.4f}\t{metrics['r2']:.4f}")
        
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model = name
    
    print("-" * 60)
    print(f"\n最佳模型: {best_model} (基于RMSE)")
    
    # 计算与最佳模型的相对性能
    print("\n与最佳模型的相对性能比较：")
    best_metrics = evaluate_model(y_true, models_dict[best_model])
    for name, pred in models_dict.items():
        if name != best_model:
            metrics = evaluate_model(y_true, pred)
            rmse_ratio = metrics['rmse'] / best_metrics['rmse']
            print(f"{name} 的RMSE是{best_model}的 {rmse_ratio:.2%}")

class WeightedEnsemble:
    """加权集成模型"""
    def __init__(self, models, n_splits=5):
        self.models = models
        self.n_splits = n_splits
        self.weights = None
        
    def _objective(self, weights, predictions, y_true):
        """优化目标函数：最小化RMSE"""
        ensemble_pred = np.zeros_like(predictions[0])
        for w, pred in zip(weights, predictions):
            ensemble_pred += w * pred
        return np.sqrt(mean_squared_error(y_true, ensemble_pred))
    
    def fit(self, X, y):
        """训练模型并优化权重"""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        all_predictions = []
        
        for model_name, model in self.models.items():
            model_predictions = np.zeros(len(X))
            for train_idx, val_idx in kf.split(X):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                model_predictions[val_idx] = model.predict(X.iloc[val_idx])
            all_predictions.append(model_predictions)
        
        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            self._objective,
            initial_weights,
            args=(all_predictions, y),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        self.weights = result.x
        print("\n模型权重：")
        for name, weight in zip(self.models.keys(), self.weights):
            print(f"{name}: {weight:.4f}")
        
        for model in self.models.values():
            model.fit(X, y)
    
    def predict(self, X):
        """使用优化后的权重进行预测"""
        predictions = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models.values()):
            predictions[:, i] = model.predict(X)
        
        ensemble_pred = np.zeros(len(X))
        for w, pred in zip(self.weights, predictions.T):
            ensemble_pred += w * pred
        
        return ensemble_pred

class StackingEnsemble:
    """Stacking集成模型"""
    def __init__(self, base_models, meta_model, n_splits=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.trained_base_models = {}
        self.trained_meta_model = None
        
    def fit(self, X, y):
        """训练stacking模型"""
        # 准备out-of-fold预测
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        # 训练基础模型并获取out-of-fold预测
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"\n训练基础模型: {name}")
            oof_predictions = np.zeros(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                print(f"训练fold {fold + 1}/{self.n_splits}")
                # 训练模型
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                # 预测验证集
                oof_predictions[val_idx] = model.predict(X.iloc[val_idx])
            
            # 存储完整的训练模型
            self.trained_base_models[name] = model
            self.trained_base_models[name].fit(X, y)
            
            # 存储out-of-fold预测
            meta_features[:, i] = oof_predictions
        
        # 训练元模型
        print("\n训练元模型...")
        self.trained_meta_model = self.meta_model
        self.trained_meta_model.mod.fit(meta_features, y)
        
        # 评估基础模型
        print("\n基础模型性能：")
        for name, model in self.trained_base_models.items():
            pred = model.predict(X)
            print(f"\n{name}模型：")
            evaluate_model(y, pred)
        
        # 评估stacking模型
        print("\nStacking模型性能：")
        meta_pred = self.predict(X)
        evaluate_model(y, meta_pred)
    
    def predict(self, X):
        """使用stacking模型进行预测"""
        # 获取基础模型的预测
        base_predictions = np.column_stack([
            model.predict(X) for model in self.trained_base_models.values()
        ])
        
        # 使用元模型进行最终预测
        return self.trained_meta_model.mod.predict(base_predictions)

# 主程序
today = feat.today_str()
train_dat = feat.get_data(split_pred=True)

# 特征工程
print("正在进行特征工程...")
train_dat = create_features(train_dat)

# 特征定义
sat_500 = ['prop_lake_500', 'r_500', 'g_500', 'b_500']
sat_1000 = ['prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']
sat_2500 = ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500']
sat_1025 = ['prop_lake_2500', 'r_2500', 'g_2500', 'b_2500', 
           'prop_lake_1000', 'r_1000', 'g_1000', 'b_1000']

# 定义基础模型
base_models = {
    'CatBoost': mod.RegMod(ord_vars=['region','cluster'],
                          dat_vars=['date'],
                          ide_vars=['latitude','longitude','maxe','dife'],
                          y='severity',
                          mod = mod.CatBoostRegressor(iterations=380, depth=6,
                             allow_writing_files=False, verbose=False)),
    
    'LightGBM': mod.RegMod(ord_vars=['region','cluster','imtype'],
                          dat_vars=['date'],
                          ide_vars=['latitude','longitude','elevation','dife'] + sat_1025,
                          y='severity',
                          mod = mod.LGBMRegressor(n_estimators=470, max_depth=8)),
    
    'XGBoost': mod.RegMod(ord_vars=['region','cluster'],
                         dat_vars=['date'],
                         y='severity',
                         mod = mod.XGBRegressor(n_estimators=70, max_depth=2))
}

# 定义元模型
meta_model = mod.RegMod(ord_vars=['region','cluster'],
                       dat_vars=['date'],
                       ide_vars=['latitude','longitude','maxe','dife'],
                       y='severity',
                       mod = mod.CatBoostRegressor(iterations=200, depth=4,
                          allow_writing_files=False, verbose=False))

# 训练基础模型
print("\n训练基础模型...")
base_predictions = {}
for name, model in base_models.items():
    print(f"\n训练{name}模型...")
    model.fit(train_dat, weight=False, cat=False)
    base_predictions[name] = model.predict(train_dat)
    print(f"{name}模型性能：")
    evaluate_model(train_dat['severity'], base_predictions[name])

# 准备元模型的特征
meta_features = pd.DataFrame(base_predictions)

# 训练元模型
print("\n训练元模型...")
meta_model.mod.fit(meta_features, train_dat['severity'])
meta_pred = meta_model.mod.predict(meta_features)

print("\nStacking模型性能：")
evaluate_model(train_dat['severity'], meta_pred)

# 预测测试集
print("\n预测测试集...")
test = feat.get_data(data_type='test')
test = create_features(test)

# 获取基础模型的预测
test_predictions = {}
for name, model in base_models.items():
    test_predictions[name] = model.predict(test)

# 准备元模型的特征
test_meta_features = pd.DataFrame(test_predictions)

# 使用元模型进行最终预测
test['pred'] = meta_model.mod.predict(test_meta_features)

form_dat = feat.sub_format(test)
print("\n预测结果分布：")
print(form_dat['severity'].value_counts())

# 检查与历史提交的相似性
mod.check_similar(form_dat)

# 与最佳提交比较
current = form_dat.copy()
mod.check_day(current,day="sub_2023_02_16.csv")
print("\n与最佳提交的差异：")
print(current.groupby('region',as_index=False)['dif_2023_02_16'].value_counts())

# 保存结果和模型
form_dat.to_csv(f'sub_STACKING_{today}.csv',index=False)
mod.save_model({'base_models': base_models, 'meta_model': meta_model}, f'mod_STACKING_{today}') 
"""
Neural network model for demand prediction
"""
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from explainer import AIExplainer

class DemandPredictor:
    def __init__(self, use_ai_explainer=False):
        """Initialize the demand prediction model."""
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.ai_explainer = AIExplainer() if use_ai_explainer else None
        self.feature_names = [
            'year', 'month', 'day', 'day_of_week',  # Time features
            'price', 'promotion_active', 'inventory_turnover',  # Market features
            'inventory_level', 'lead_time_days', 'fulfillment_rate'  # Supply chain features
        ]
        
    def _build_model(self):
        """Build the neural network model architecture."""
        inputs = tf.keras.Input(shape=(10,))  # 10 features
        
        # Time features branch
        time_features = tf.keras.layers.Lambda(lambda x: x[:, :4])(inputs)
        time_dense = tf.keras.layers.Dense(32, activation='relu')(time_features)
        
        # Market features branch
        market_features = tf.keras.layers.Lambda(lambda x: x[:, 4:7])(inputs)
        market_dense = tf.keras.layers.Dense(32, activation='relu')(market_features)
        
        # Supply chain features branch
        supply_features = tf.keras.layers.Lambda(lambda x: x[:, 7:])(inputs)
        supply_dense = tf.keras.layers.Dense(32, activation='relu')(supply_features)
        
        # Combine branches
        combined = tf.keras.layers.Concatenate()([time_dense, market_dense, supply_dense])
        
        # Deep layers
        x = tf.keras.layers.Dense(256, activation='relu')(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Custom loss function that penalizes underprediction more
        def asymmetric_loss(y_true, y_pred):
            error = y_true - y_pred
            return tf.reduce_mean(tf.where(error > 0, error * 4, -error))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=asymmetric_loss,
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data):
        """Prepare features and target from raw data."""
        if isinstance(data, pd.DataFrame):
            # If DataFrame contains date column, extract features
            if 'date' in data.columns:
                data['year'] = data['date'].dt.year
                data['month'] = data['date'].dt.month
                data['day'] = data['date'].dt.day
                data['day_of_week'] = data['date'].dt.dayofweek
            
            # Select features in correct order
            feature_columns = [
                'year', 'month', 'day', 'day_of_week',  # Time features
                'price', 'promotion_active', 'inventory_turnover',  # Market features
                'inventory_level', 'lead_time_days', 'fulfillment_rate'  # Supply chain features
            ]
            
            # For prediction data, some columns might be missing
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].values
            
            # If target is available
            y = data['demand'].values if 'demand' in data.columns else None
        else:
            # Assume numpy array with features only
            X = data
            y = None
            
        # Scale features
        X_scaled = self.scaler.transform(X) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, data, epochs=200, validation_split=0.2):
        """Train the model on the provided data."""
        X, y = self.prepare_data(data)
        
        # Callbacks for training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-6
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def predict(self, data):
        """Make predictions on new data."""
        if isinstance(data, pd.DataFrame):
            X_scaled, _ = self.prepare_data(data)
        else:
            X_scaled = self.scaler.transform(data)
        return self.model.predict(X_scaled).flatten()
    
    def evaluate(self, data):
        """Evaluate the model on test data."""
        X, y = self.prepare_data(data)
        metrics = self.model.evaluate(X, y, verbose=0)
        
        # Make predictions
        y_pred = self.predict(data)
        
        # Calculate additional metrics
        rmse = np.sqrt(metrics[1])
        
        results = {
            'loss': metrics[0],
            'mae': metrics[1],
            'rmse': rmse,
            'predictions': [
                {
                    'actual': float(y),
                    'predicted': float(y_hat),
                    'difference': float(y - y_hat)
                }
                for y, y_hat in zip(y[:5], y_pred[:5])  # Show first 5 predictions as examples
            ]
        }
        
        # Get model explanations
        if hasattr(self, 'model_explainer'):
            try:
                # Calculate SHAP values for the test set
                shap_values = self.model_explainer.get_shap_values(X)
                
                # Get feature importance plot
                self.model_explainer.plot_feature_importance(shap_values)
                
                # Get detailed explanations for sample predictions
                sample_explanations = []
                for i in range(min(5, len(X))):
                    explanation = self.model_explainer.explain_prediction(
                        X[i],
                        shap_values=shap_values[i]
                    )
                    sample_explanations.append(explanation)
                
                results['explanations'] = {
                    'shap_values': shap_values.tolist(),
                    'sample_explanations': sample_explanations
                }
                
            except Exception as e:
                results['explanation_error'] = str(e)
        
        # Get AI explanations if enabled
        if self.ai_explainer:
            try:
                import asyncio
                explanations = asyncio.run(self.ai_explainer.get_combined_explanation(results))
                results['ai_explanations'] = explanations
            except Exception as e:
                results['ai_explanations_error'] = str(e)
        
        return results
    
    def get_feature_importance(self, X):
        """Get feature importance using gradient-based method."""
        if isinstance(X, pd.DataFrame):
            X, _ = self.prepare_data(X)
            
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X)
            predictions = self.model(X)
        
        gradients = tape.gradient(predictions, X)
        importance = tf.reduce_mean(tf.abs(gradients), axis=0)
        return {name: float(imp) for name, imp in zip(self.feature_names, importance.numpy())}

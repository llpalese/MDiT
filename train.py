# Copyright (c) 2026 University of Bari.
# Licensed under GNU AGPLv3 OR Commercial License. See LICENSE file for details.


import os
import numpy as np
import json
import argparse

# Configure matplotlib for headless use (no GUI) before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Forced JAX backend
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import ops

# --- DATA GENERATOR (Modular Version) ---

def make_data_generator(file_list, lookback, forecast, batch_size, shuffle=True):
    """
    Creates a generator from a specified list of .npz files.
    """
    if not file_list:
        return None, 0, 0

    trajectories = {}
    data_indices = []
    
    for f in file_list:
        data = np.load(f)['coords'].astype('float32')
        # Normalizzazione Z-score
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-7
        trajectories[f] = (data - mean) / std
        
        n_frames = len(data)
        for i in range(n_frames - lookback - forecast):
            data_indices.append((f, i))
    
    data_indices = np.array(data_indices, dtype=object)
    num_samples = len(data_indices)
    steps_per_epoch = int(np.ceil(num_samples / batch_size))
    
    def generator():
        while True:
            indices = np.arange(num_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                batch_x, batch_y = [], []
                
                for idx in batch_idx:
                    f_path, f_start = data_indices[idx]
                    traj = trajectories[f_path]
                    batch_x.append(traj[f_start : f_start + lookback])
                    batch_y.append(traj[f_start + lookback : f_start + lookback + forecast])
                
                yield np.array(batch_x), np.array(batch_y)
                
    return generator(), steps_per_epoch, trajectories[file_list[0]].shape[1]

# --- CUSTOM LOSS: MSE + RMSF PENALTY ---
def rmsf_weighted_mse_loss(y_true, y_pred, alpha=0.1):
    """
    Hybrid loss: standard MSE + fluctuation difference penalty (RMSF-like).
    - y_true, y_pred shape: (batch_size, forecast, num_variates)
    """
    # 1. MSE Classic
    mse_loss = ops.mean(ops.square(y_true - y_pred))
    
    # 2. Temporal fluctuation (RMSF proxy calculated along the forecast/time axis)
    rmsf_true = ops.std(y_true, axis=1)
    rmsf_pred = ops.std(y_pred, axis=1)
    
    # We penalize the deviation between the fluctuations
    rmsf_loss = ops.mean(ops.square(rmsf_true - rmsf_pred))
    
    # Combined loss (alpha controls the weight of the fluctuation)
    return mse_loss + alpha * rmsf_loss

# --- ITRANSFORMER BLOCK ---
class iTransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.layernorm1 = keras.layers.LayerNormalization()
        self.layernorm2 = keras.layers.LayerNormalization()
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="gelu"),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout)
        ])

    def call(self, x):
        attn_out = self.attention(x, x)
        x = self.layernorm1(x + attn_out)
        x = self.layernorm2(x + self.ffn(x))
        return x

def build_itransformer(lookback, num_variates, forecast, d_model=128, num_heads=8, ff_dim=512):
    inputs = keras.layers.Input(shape=(lookback, num_variates), name="Input_Trajectories")
    x = keras.layers.Permute((2, 1), name="Inversion_Time_Vars")(inputs)
    x = keras.layers.Dense(d_model, name="Variable_Embedding")(x)
    x = iTransformerBlock(d_model, num_heads, ff_dim, name="iTransformer_Core")(x)
    x = keras.layers.Dense(forecast, name="Temporal_Projection")(x)
    outputs = keras.layers.Permute((2, 1), name="Output_Trajectories")(x)
    return keras.Model(inputs, outputs, name="iTransformer_3QL3")

# --- MAIN LOGIC ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iTransformer Training with Curriculum Learning and RMSF Loss")
    parser.add_argument("--train_dir", type=str, required=True, help="Training data folder")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation data folder")
    parser.add_argument("--test_file", type=str, required=True, help="Single .npz file for final testing")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--forecast", type=int, default=50, help="Desired final forecast horizon")
    parser.add_argument("--initial_forecast", type=int, default=10, help="Starting horizon for Curriculum Learning")
    parser.add_argument("--forecast_increment", type=int, default=20, help="Forecast increment step per phase")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for each phase of Curriculum Learning")
    args = parser.parse_args()

    # 1. Preparing file lists
    train_files = [os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith('.npz')]
    val_files = [os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir) if f.endswith('.npz')]
    test_files = [args.test_file]

    # We determine the number of variables from the data
    temp_data = np.load(train_files[0])['coords'].astype('float32')
    num_vars = temp_data.shape[1]

    # Curriculum Learning Management: defining the sequence of increasing forecasts
    current_forecast = args.initial_forecast
    forecast_stages = []
    while current_forecast < args.forecast:
        forecast_stages.append(current_forecast)
        current_forecast += args.forecast_increment
    forecast_stages.append(args.forecast) # 

    print("\n" + "="*60)
    print(f"LEARNING CURRICULUM PLAN - Planned phases: {forecast_stages}")
    print("="*60)

    # Let's build the basic model
    model = build_itransformer(args.lookback, num_vars, args.forecast)
    weights_temp_path = "curriculum_temp_weights.weights.h5"
    
    all_histories = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    # Curriculum Learning Cycle
    for stage_idx, f_steps in enumerate(forecast_stages):
        print(f"\n--- PHASE {stage_idx + 1}/{len(forecast_stages)}: Forecast Horizon = {f_steps} ---")
        
        # Generators calibrated to the current horizon of this phase
        # Fixed generators on the maximum horizon, while the curriculum acts via loss
        train_gen, train_steps, _ = make_data_generator(train_files, args.lookback, args.forecast, args.batch_size)
        val_gen, val_steps, _ = make_data_generator(val_files, args.lookback, args.forecast, args.batch_size, shuffle=False)

        total_train_steps = args.epochs * train_steps

        # Optimizer and LR scheduler for the phase
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=total_train_steps,
            alpha=0.1
        )
        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-3)

        # Compilation (Note: the final Dense layer has a fixed args.forecast size, but slicing/loss affects f_steps)
        # For simplicity in the curriculum, we use the full output model, masking or trimming batches if necessary,
        # or we use the architecture directly. Here we pass the custom loss.
        model.compile(optimizer=optimizer, loss=lambda y_true, y_pred, fs=f_steps: rmsf_weighted_mse_loss(y_true[:, :fs, :], y_pred[:, :fs, :], alpha=2.0), metrics=None) # Avoid shape conflicts on metrics 

       # Curriculum Learning cycle
    for stage_idx, f_steps in enumerate(forecast_stages):
        print(f"\n--- PHASE {stage_idx + 1}/{len(forecast_stages)}: Active Forecast Horizon = {f_steps} (out of {args.forecast} total) ---")
        
        # We always use the maximum horizon in the generator to keep the shapes consistent with the model
        train_gen, train_steps, _ = make_data_generator(train_files, args.lookback, args.forecast, args.batch_size)
        val_gen, val_steps, _ = make_data_generator(val_files, args.lookback, args.forecast, args.batch_size, shuffle=False)

        total_train_steps = args.epochs * train_steps

        # Optimizer and LR scheduler for the phase
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=total_train_steps,
            alpha=0.1
        )
        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-3)

        # Compilation: We slice both y_true and y_pred to the current number of steps (fs=f_steps)
        model.compile(
            optimizer=optimizer, 
            loss=lambda y_true, y_pred, fs=f_steps: rmsf_weighted_mse_loss(y_true[:, :fs, :], y_pred[:, :fs, :], alpha=0.3), 
            metrics=None
        )

        if stage_idx > 0 and os.path.exists(weights_temp_path):
            print("Loading weights from the previous phase (Warm Start)...")
            model.load_weights(weights_temp_path)

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=weights_temp_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss"
        )

        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=args.epochs,
            callbacks=[checkpoint, keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )

        # We accumulate metrics for the overall plot
        all_histories["loss"].extend(history.history["loss"])
        all_histories["val_loss"].extend(history.history["val_loss"])

    # Final saving of optimal weights
    model.save_weights("best_itransformer_weights_rmsf_loss.weights.h5")

    # 6. Saving the complete history and generating the plot
    with open("training_history_rmsf_loss.json", "w") as f:
        json.dump(all_histories, f)

    # Generating and saving the error graph (Training vs. Validation)
    print("\nGenerating the error graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(all_histories["loss"], label="Training Loss (RMSF-Weighted MSE)", color="blue", linewidth=2)
    plt.plot(all_histories["val_loss"], label="Validation Loss", color="orange", linestyle="--", linewidth=2)
    
    # Vertical lines to highlight the phase transitions of the Curriculum Learning
    epochs_per_stage = args.epochs
    for i in range(1, len(forecast_stages)):
        plt.axvline(x=i * epochs_per_stage, color="gray", linestyle=":", alpha=0.7, label="Curriculum Phase Change" if i == 1 else "")

    plt.xlabel("Total Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Learning Curve (Curriculum Learning + RMSF Loss)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plot_filename = "training_validation_loss_plot.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph successfully saved as: '{plot_filename}'")

    print("\nTraining with Curriculum Learning successfully completed.")

    print("\n--- Final Evaluation on Test Set---")
    # Let's recreate the test generator with the full final horizon
    test_gen, test_steps, _ = make_data_generator(test_files, args.lookback, args.forecast, args.batch_size, shuffle=False)
    
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: rmsf_weighted_mse_loss(y_true, y_pred, alpha=0.3), metrics=["mae"])
    model.load_weights("best_itransformer_weights_rmsf_loss.weights.h5")
    test_results = model.evaluate(test_gen, steps=test_steps)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test MAE: {test_results[1]:.4f}")

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train(autoencoder, x_train_noisy, x_train, x_test_noisy, x_test, epochs=50, batch_size=128, model_save_path='best_model.h5'):
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Model checkpoint callback to save the best model
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

    # Train the model
    history = autoencoder.fit(
        x_train_noisy, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    return history

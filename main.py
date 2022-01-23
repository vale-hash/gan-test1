from keras.models import Sequential
from keras.layers import Input, Reshape , Dropout ,Dense ,Flatten,BatchNormalization,Activation,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
import numpy as np
from PIL import Image
import os
#preview image frame
preview_rows=4
preview_column=7
preview_margin=4
save_freq=100
#noise: random variable fed in
noise_size = 100
#configuration/model fitting 
EPOCHS=10000
batch_size = 32
Genreate_res =3
Image_Size=200
image_channels=3
def discriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = 3, strides=2,input_shape = image_shape, padding ='same'))
    model.add(LeakyReLU(alpha= 0.2))
    model.add (Dropout(0.25))

#second covolution layer
    model.add(Conv2D(64,kernel_size=3, stride=2 ,padding = same))
    model.add(ZeroPadding2D(padding=((0,1), (0,1))))
    model.add(BatchNormalization (momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

#3rd convo layer
    model.add(Conv2D(128, kernel_size=3, strides=2, padding= same))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
#layer 4
    model.add(Conv2D(256, kernel_size=3, strides=1, padding= same))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
#layer5
    model.add(Conv2D(256, kernel_size=3, strides=1, padding= same))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    input_image=Input(shape =image_shape)
    validity=model(input_image)
    return(input_image, validity)
#building the generator
def generator(noise_size,channels):
    model=Sequential()
    model.add(Dense(4*4*256,activation='relu',input_dim=noise_size))
    model.add(Reshape((4,4,256)))
    
    #2nd layer
    model.add(UpSampling2D())
    model.add(Conv2D(256 , kernel_size =3 , padding = same))
    model.add(BatchNormalization(momentum= 0.8))
    model.add(Activation('relu'))
    #3rd layer
    model.add(UpSampling2D())
    model.add(Conv2D(256 , kernel_size =3 , padding = same))
    model.add(BatchNormalization(momentum= 0.8))
    model.add(Activation('relu'))
    
    for i in  range (GENERATE_RES):
        model.add(UpSampling2D())
        model.add(Covo2D(256 , kernel_size = 3 , padding = same))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation('relu'))
    model.summary()
    model.add(Conv2D(channels, kernel_size =3, padding = same))
    model.add(Activation('tanh'))
    
    input_im=Input(Shape=(noise_size,))
    generated_image=model(input_im)
    return Model(input_im,generated_image)
#function that takes count and noise as input , genrates frames from parameters and saves
def save_images(cnt, noise):
    image_array = np.full((
        preview_margins + (preview_rows * (Image_Size + preview_margin)),
        preview_margin + (preview_column * (Image_Size + preview_margin)), 3),255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(preview_rows):
        for col in range(preview_column):
            r = row * (Image_Size + preview_margin) + preview_margin
            c = col * (Image_Size + preview_margin) + preview_margin
            image_array[r:r + Image_Size, c:c +
                        Image_Size] = generated_images[image_count] * 255
            image_count += 1
    output_path = 'filepath'
    results_gan=os.path.join(output_path, f'trained-{cnt}.png')
   
    im = Image.fromarray(image_array)
    im.save(results_gan)
    #compile and train models
image_shape =(Image_Size,Image_Size,image_channels)

active_discriminator= discriminator(image_shape)
active_discriminator.compile(loss='binary cross entropy',optimizer='adam' , metrics=['accuracy'])

#generator
active_generator= generator(noise_size,image_channels)
noise_input=Input(shape=(noise_size))
generated_image = active_generator(noise_input)

active_discriminator.trainable = False
validity=active_discriminator(generated_image)

combined=Model(noise_input,validity)
combined.compile(loss = ' binary_crossentropy', optimizer= 'Adams', metrocs=['accuracy'])

y_real = np.ones((batch_size,1))
x_real = np.ones((batch_size,1))

fixed_noise= np.random.normal(0,1,(preview_rows*preview_column,noise_size))

cnt=1
for epoch in range (EPOCHS):
    idx=np.random.randint(0,training_data.shape[0],batch_size)
    x_real = training_data[idx]
    
    noise =np.random.normal(0,1,(batch_size,noise_size))
    x_fake=active_generator.predict(noise)
    
    discriminator_metric_real=active_discriminator.train_on_batch(x_real,y_real)
    
    discriminator_metric_generated=active_discriminator.train_on_batch(x_fake,y_fake)
    
    discriminator_metric=0.5*np.add(discriminator_metric_real,discriminator_metric_generated)
    
    generateed_metric = combined.train_on_batch(noise,y_real)
    if epoch % save_freq==0:
        save_images(cnt,fixed_noise)
        cnt +=1
        
        print(f'{epoch} epoch, discriminator accuracy : {100*discriminator_metric[1]} , Generator accuracy:{100*generator_metric[1]
})

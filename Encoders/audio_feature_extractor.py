import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" # set gpu number
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
import h5py

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'
num_secs = 10 # length of the audio sequence. Videos in our dataset are all 10s long.
freq = 1000
sr = 44100

# path of audio files and AVE annotation
audio_dir = "/home/liuxiangxi/Encoders/data/AVE_audio/AVE_wav" # .wav audio files
lis = os.listdir(audio_dir)
len_data = len(lis)
audio_features = np.zeros([len_data, 10, 128])

i = 0
for n in range(len_data):

    '''feature learning by VGG-net trained by audioset'''
    audio_index = os.path.join(audio_dir, lis[n]) # path of your audio files

    input_batch = vggish_input.wavfile_to_examples(audio_index)
    np.testing.assert_equal(
        input_batch.shape,
        [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

    # Define VGGish, load the checkpoint, and run the batch through the model to
    # produce embeddings.
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: input_batch})
        #print('VGGish embedding: ', embedding_batch[0])
        #print(embedding_batch.shape)
        audio_features[i, :, :] = embedding_batch
        i += 1
        print(i)


# save the audio features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file

with h5py.File('/home/liuxiangxi/Encooders/data/audio_embedding.h5', 'w') as hf:
    hf.create_dataset("dataset",  data=audio_features)

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 02:38:40 2018

@author: Edwin
"""

""" Implementation in TensorFlow of the paper 
A Neural Algorithm of Artistic Style (Gatys et al., 2016) 
Modified by Edwin Yuan (edyuan@stanford.edu)
base-line code by Chip Huyen (chiphuyen@cs.stanford.edu) 
CS20: "TensorFlow for Deep Learning Research"
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import tensorflow as tf

import load_vgg
import utils

def setup():
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('outputs')

class StyleTransfer(object):
    def __init__(self, content_img, style_img, img_width, img_height):
        '''
        img_width and img_height are the dimensions we expect from the generated image.
        We will resize input content image and input style image to match this dimension.
        Feel free to alter any hyperparameter here and see how it affects your training.
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.content_img = utils.get_resized_image(content_img, img_width, img_height)
        self.style_img = utils.get_resized_image(style_img, img_width, img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_height)

        ###############################
        ## create global step (gstep) and hyperparameters for the model
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        # content_w, style_w: corresponding weights for content loss and style loss
        self.content_w = 10e-6
        self.style_w = 50*10e-6
        # style_layer_w: weights for different style layers. deep layers have more weights
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0] 
        self.gstep = tf.Variable(0,trainable=False,name ='global_step')# global step
        self.lr = 2.0
        ###############################

    def create_input(self):
        '''
        We will use one input_img as a placeholder for the content image, 
        style image, and generated image, because:
            1. they have the same dimension
            2. we have to extract the same set of features from them
        We use a variable instead of a placeholder because we're, at the same time, 
        training the generated image to get the desirable result.
        Note: image height corresponds to number of rows, not columns.
        '''
        with tf.variable_scope('input') as scope:
            self.input_img = tf.get_variable('in_img', 
                                        shape=([1, self.img_height, self.img_width, 3]),
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
    def load_vgg(self):
        '''
        Load the saved model parameters of VGG-19, using the input_img
        as the input to compute the output at each layer of vgg.
        During training, VGG-19 mean-centered all images and found the mean pixels
        to be [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract
        this mean from our images.
        '''
        self.vgg = load_vgg.VGG(self.input_img)
        self.vgg.load()
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def _content_loss(self, P, F):
        ''' Calculate the loss between the feature representation of the
        content image and the generated image.
        
        Inputs: 
            P: content representation of the content image
            F: content representation of the generated image
            Read the assignment handout for more details
            Note: Don't use the coefficient 0.5 as defined in the paper.
        '''
        ###############################
        # Content loss from L1 Norm
        # F - gen img content has shape (1,32,42,512) tf.tensor
        # P - flaot32 array (1,32,42,512)
        Pshape = P.shape
        coefficient = 1./(4.0*np.prod(Pshape))
        
        content_diff = F-P
        self.content_loss = coefficient * tf.reduce_sum(tf.multiply(content_diff,content_diff))
        #self.content_loss = tf.Print(self.content_loss, [self.content_loss])
        ###############################
        
    def _gram_matrix(self, F, N, M):
        """ Create and return the gram matrix for tensor F
        """
        #numfilters = F.shape.as_list()[3]
        F= tf.reshape(F,shape=[M,N])
        
        return tf.matmul(tf.transpose(F),F)

    def _single_style_loss(self, a, g):
        """ Calculate the style loss at a certain layer
        Inputs:
            a is the feature representation of the style image at that layer
            g is the feature representation of the generated image at that layer
        Output:
            the style loss at a certain layer (which is E_l in the paper)
        Hint: 1. you'll have to use the function _gram_matrix()
            2. we'll use the same coefficient for style loss as in the paper
            3. a and g are feature representation, not gram matrices
        """
        ###############################
        # Compute difference of gram matrices as style loss
        # a- numpy array (1,n,m,512)
        # g-tensorflow array (1,n,m,512)
        N = a.shape[3]
        M = a.shape[2]*a.shape[1]
        a = tf.constant(a)
        style_difference = self._gram_matrix(a,N,M) - self._gram_matrix(g,N,M)
        out = tf.reduce_sum(tf.multiply(style_difference,style_difference))
        out *= 1/(4*(N**2)*(M**2))
        #out = tf.Print(out, [out])   
        return out
        ###############################

    def _style_loss(self, A):
        """ Calculate the total style loss as a weighted sum 
        of style losses at all style layers
        Hint: you'll have to use _single_style_loss()
        """
        ###############################
        ## Wrapper function which calls '_single_style_loss' on each layer
        # A - list of 5 numpy arrays, arrays have different sizes but (1,x,y,512)
        # g- gen img tensorflow array, (1,32,42,512)
        self.style_loss = 0
        w = self.style_layer_w
        for ind,layer in enumerate(self.style_layers):
            gen_img_content = getattr(self.vgg, layer)
            self.style_loss += w[ind]*self._single_style_loss(A[ind],gen_img_content)
        ###############################

    def losses(self):
        with tf.variable_scope('losses') as scope:
            with tf.Session() as sess:
                # assign content image to the input variable
                #sess.run(tf.global_variables_initializer())
                sess.run(self.input_img.assign(self.content_img)) 
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            with tf.Session() as sess:
                #sess.run(tf.global_variables_initializer())
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            self._style_loss(style_layers)

            ##########################################
            ## Total Loss Function
            self.total_loss = self.style_w* self.style_loss + self.content_w*self.content_loss
            ##########################################

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss,global_step=self.gstep)

    def create_summary(self):
        ## TO DO: create summaries for all the losses
        ## Hint: don't forget to merge them
        with tf.name_scope('summaries'):
            tf.summary.scalar('total loss', self.total_loss)
            tf.summary.scalar('style loss', self.style_loss)
            tf.summary.scalar('content loss', self.content_loss)
            tf.summary.histogram('histogram loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()


    def build(self):
        self.create_input()
        self.load_vgg()
        self.losses()
        self.optimize()
        self.create_summary()

    def train(self, n_iters):
        skip_step = 1
        with tf.Session() as sess:
            
            ###############################
            ## TO DO: 
            ## Sess.run and FileWriter functions
            ###############################
            #sess.run()
            sess.run(tf.global_variables_initializer())
            sess.run(self.input_img.assign(self.initial_img))
            
            writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

            ###############################
            ## Restore Model
            ##############################
            saver = tf.train.Saver()
            if os.path.isfile('/tmp/model.ckpt'):
                saver.restore(sess, "/tmp/model.ckpt")

            initial_step = self.gstep.eval()
            
            start_time = time.time()
            for index in range(initial_step, n_iters):
                print (index)
                if index >= 5 and index < 20:
                    skip_step = 10
                elif index >= 20:
                    skip_step = 20
                
                _,contentloss,styleloss = sess.run([self.opt,self.content_loss,self.style_loss])
                if (index + 1) % skip_step == 0:
                    ###############################
                    ##obtain generated image, loss, and summary
                   # gen_img_content = getattr(self.vgg, self.content_layer)
                    gen_image, total_loss= sess.run([self.input_img,self.total_loss])
                    #contentloss = self.content_loss
                    summary = sess.run(self.summary_op)
                    ###############################
                    
                    # add back the mean pixels we subtracted before
                    gen_image = gen_image + self.vgg.mean_pixels 
                    writer.add_summary(summary, global_step=index)
                    print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                    print('   Loss: {:5.1f}'.format(total_loss))
                    print('   Content Loss: {:5.1f}'.format(contentloss))
                    print('   Style Loss: {:5.1f}'.format(styleloss))
                    print('   Took: {} seconds'.format(time.time() - start_time))
                    start_time = time.time()

                    filename = 'outputs/%d.png' % (index)
                    utils.save_image(filename, gen_image)

                    if (index + 1) % 1 == 0:
                        ###############################
                        ## save the variables into a checkpoint
                        save_path = saver.save(sess, "./ckpt/model.ckpt")
                        ###############################
                        pass

if __name__ == '__main__':
    setup()
    machine = StyleTransfer('content/motor.jpg', 'styles/orante.jpg', 333, 250)
    machine.build()
    machine.train(300)
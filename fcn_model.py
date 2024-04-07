import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG-16 and use its features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # Encoder
        self.features_block1 = nn.Sequential(*features[:5])  # First pooling
        self.features_block2 = nn.Sequential(*features[5:10])  # Second pooling
        self.features_block3 = nn.Sequential(*features[10:17])  # Third pooling
        self.features_block4 = nn.Sequential(*features[17:24])  # Fourth pooling
        self.features_block5 = nn.Sequential(*features[24:])  # Fifth pooling

        # Modify the classifier part of VGG-16
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Decoder
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        #raise NotImplementedError("Implement the forward method")

        _,_,i0,j0=x.size()

        x1 = self.features_block1(x)
        x2 = self.features_block2(x1)
        x3 = self.features_block3(x2)
        x4 = self.features_block4(x3)
        x5 = self.features_block5(x4)

        # Classifier
        score = self.classifier(x5)

        # Decoder
        upscore2 = self.upscore2(score) # need to crop this
        score_pool4 = self.score_pool4(x4)
        # Crop to ensure consistent resolution before combining

        #upscore2=upscore2[:, :, 1:25, 1:33]

        _,_,i1,j1=score_pool4.size()
        _,_,i11,j11=upscore2.size()

        dif1=int((i11-i1)/2)
        dif11=int((j11-j1)/2)

        upscore2=upscore2[:, :, dif1:i1+dif1, dif11:j1+dif11]
        

        #print(f"NOW The size of upscore2 is (need crop) {upscore2.size()}")
        #print(f"The size of score_pool4 is {score_pool4.size()}")


        #score_pool4 = score_pool4[:, :, ?:? + upscore2.size()[2], 2:2 + upscore2.size()[3]] # this is cropping upsampling
        # we want to crop upscore2 instead

        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2) # crop this one

        score_pool3 = self.score_pool3(x3)
        # Crop to ensure consistent resolution before combining
        #score_pool3 = score_pool3[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]] # not crop this one 

        #upscore_pool4=upscore_pool4[:,:,1:49,1:65] 

        _,_,i2,j2=score_pool3.size()
        _,_,i22,j22=upscore_pool4.size()

        dif2=int((i22-i2)/2)
        dif22=int((j22-j2)/2)

        upscore_pool4=upscore_pool4[:, :, dif2:i2+dif2, dif22:j2+dif22]

        #print(f"The size of upscore_pool4 is (need crop) {upscore_pool4.size()}")
        #print(f"The size of score_pool3 is {score_pool3.size()}")

        upscore_final = self.upscore_final(score_pool3 + upscore_pool4)

        _,_,i00,j00=upscore_final.size()

        dif0=int((i00-i0)/2)
        dif00=int((j00-j0)/2)



        #print(f"before: Size of upscore_final{upscore_final.size()}")
        upscore_final=upscore_final[:,:,dif0:i0+dif0,dif00:j0+dif00]
        #print(f"after: Size of upscore_final{upscore_final.size()}")

        # Return final prediction
        return upscore_final



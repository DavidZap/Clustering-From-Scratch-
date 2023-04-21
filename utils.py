from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Utils:

    def image_convert(self,filename,Originalfolder,FinalFolder):
        import matplotlib.image as mpimg
        if Originalfolder == '':
            img=Image.open(os.path.join(Originalfolder, filename))
            img = img.convert("L")
            img = img.resize((256, 256))
            img = np.array(img)
            return img
        else:
            img=Image.open(os.path.join(Originalfolder, filename))
            img = img.convert("L")
            img = img.resize((256, 256))
            img.save(os.path.join(FinalFolder, filename))
            img = mpimg.imread(os.path.join(FinalFolder, filename))
            return img
        
    def plotv1(self,X_transformed,labels,y):

        import matplotlib.pyplot as plt

        plot = plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y)

        plt.legend(handles=plot.legend_elements()[0], 
                   labels=labels)

        return plt.show()
    
    def plot_ss(self,X,n_components,method,labels,y):
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        from model import myPCA
        myPCA = myPCA(n_components=n_components,method=method)

        # Normalise the data
        scaler = StandardScaler()
        scaler.fit(X)
        X_normalised = scaler.fit_transform(X)

        # Apply PCA now
        myPCA.fit(X_normalised)

        # transform the data using the PCA object
        X_transformed = myPCA.fit_transform(X_normalised)


        plot = plt.scatter(X_transformed[:,0], X_transformed[:,1], c=y)

        plt.legend(handles=plot.legend_elements()[0], 
                   labels=labels)

        plt.show()
    



        
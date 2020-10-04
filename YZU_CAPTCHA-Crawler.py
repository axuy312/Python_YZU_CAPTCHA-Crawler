import matplotlib, requests, numpy, time, cv2, os, shutil, PIL, random
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


basewidth = 50

#init-----------------------------
print('正在初始化.....')
digits = []
labels = []
basewidth = 50
fig = plt.figure(figsize = (20,20))
cnt = 0 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(0,10)
    for img in os.listdir('{}'.format(i))
        pil_image = PIL.Image.open('{}{}'.format(i, img)).convert('1') 

        wpercent = (basewidthfloat(pil_image.size[0]))
        hsize = int((float(pil_image.size[1])float(wpercent)))
        img = pil_image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)

        ax = fig.add_subplot(40, 16, cnt+1, xticks=[], yticks=[])
        ax.imshow(img,cmap=plt.cm.binary,interpolation='nearest')
        ax.text(0, 7, str(i), color=red, fontsize = 20)        
        cnt = cnt + 1

        digits.append([pixel for pixel in iter(img.getdata())])
        labels.append(i)
digit_ary = numpy.array(digits)
scaler = StandardScaler()
scaler.fit(digit_ary)
X_scaled = scaler.transform(digit_ary)
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='logistic', max_iter = 10000)
mlp.fit(X_scaled,labels)
joblib.dump(mlp, '.pklcaptcha.pkl')
clf = joblib.load('.pklcaptcha.pkl')
#end init--------------------------------

def savepkl()
    joblib.dump(mlp, '.pklcaptcha.pkl')
def loadpkl()
    clf = joblib.load('.pklcaptcha.pkl')


def saveKaptcha(image, dest)
    print('saveKaptcha')
    scaler = StandardScaler()
    pil_image = PIL.Image.open(image).convert('RGB')
    open_cv_image = numpy.array(pil_image) 
    imgray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda xx[1])
    ary = []
    aryall = []
    cnt = 0
    for (c,_) in cnts
        (x,y,w,h) = cv2.boundingRect(c)
        print((x,y,w,h))
        if h = 30 and w  30
            cnt = cnt + 1
            ary.append((x,y,w,h))
        aryall.append((x,y,w,h))
    if(cnt != 4)
        print(aryall)
    data = []
    for idx, (x,y,w,h) in enumerate(ary)
        fig = plt.figure()
        roi = open_cv_image[yy+h, xx+w]
        thresh = roi.copy()
        plt.imshow(thresh)
        #print(thresh)
        #print(os.path.join(dest, '{}.jpg'.format(idx)));
        plt.savefig(os.path.join(dest, '{}.jpg'.format(idx)), dpi=100)

def predictKaptcha(dest)
    print('predictKaptcha')
    data = []
    for idx, img in enumerate(os.listdir(dest))
        pil_image = PIL.Image.open(os.path.join(dest,'{}'.format(img))).convert('1') 
        wpercent = (basewidthfloat(pil_image.size[0]))
        hsize = int((float(pil_image.size[1])float(wpercent)))
        img = pil_image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)
        data.append([pixel for pixel in iter(img.getdata())])
    scaler = StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    print(data_scaled)
    return clf.predict(data_scaled)

def train()
    print('train')
    rs  = requests.session()

    headers = {'user-agent' 'Mozilla5.0 (Macintosh Intel Mac OS X 10_13_4) AppleWebKit537.36 (KHTML, like Gecko) Chrome66.0.3359.181 Safari537.36'}
    
    veriurl = 'httpsportalx.yzu.edu.twPortalSocialVBSelRandomImage.aspxUID=WQBaADIAMAAyADAALwAwADcALwAxADAAcABvAHIAdABhAGwAeAA%3d'
    res = rs.get('httpsportalx.yzu.edu.twPortalSocialVBLogin.aspx', headers=headers)
    with open('kaptcha.jpg', 'wb') as f
        res2 = rs.get(veriurl, headers=headers)
        f.write(res2.content)
    saveKaptcha('kaptcha.jpg', 'tmp') 
    kaptcha = predictKaptcha('tmp')

    print(kaptcha)
    return kaptcha

def generateModel()
    print('generateModel')
    digits = []
    labels = []
    basewidth = 50
    fig = plt.figure(figsize = (20,20))
    cnt = 0 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(0,10)
        for img in os.listdir('{}'.format(i))
            pil_image = PIL.Image.open('{}{}'.format(i, img)).convert('1') 

            wpercent = (basewidthfloat(pil_image.size[0]))
            hsize = int((float(pil_image.size[1])float(wpercent)))
            img = pil_image.resize((basewidth,hsize), PIL.Image.ANTIALIAS)

            ax = fig.add_subplot(40, 16, cnt+1, xticks=[], yticks=[])
            ax.imshow(img,cmap=plt.cm.binary,interpolation='nearest')
            ax.text(0, 7, str(i), color=red, fontsize = 20)        
            cnt = cnt + 1

            digits.append([pixel for pixel in iter(img.getdata())])
            labels.append(i)
    digit_ary = numpy.array(digits)
    scaler = StandardScaler()
    scaler.fit(digit_ary)
    X_scaled = scaler.transform(digit_ary)
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='logistic', max_iter = 10000)
    mlp.fit(X_scaled,labels)

 
    

#generateModel()
#savepkl()
#loadpkl()
while True
    kap = train()
    imgplot = plt.imshow(matplotlib.image.imread('kaptcha.jpg'))
    plt.ion()
    plt.show()
    print(kap)
    correct = input(正確 )
    if correct == '1'
        print('wait...')
        ran = str(random.randint(1,10000000))
        for i in range(0,4)
            os.rename(os.path.join('tmp', str(i)+'.jpg'), os.path.join('tmp', ran+str(i)+'.jpg'))
            shutil.move('.tmp'+ran+str(i)+'.jpg','.'+str(kap[i]))
        #generateModel()
        #savepkl()
        #loadpkl()
    elif correct == '0'
        kap[0]=input(更正+str(kap[0])+-)
        kap[1]=input(更正+str(kap[1])+-)
        kap[2]=input(更正+str(kap[2])+-)
        kap[3]=input(更正+str(kap[3])+-)
        print('wait...')
        ran = str(random.randint(1,10000000))
        for i in range(0,4)
            os.rename(os.path.join('tmp', str(i)+'.jpg'), os.path.join('tmp', ran+str(i)+'.jpg'))
            shutil.move('.tmp'+ran+str(i)+'.jpg','.'+str(kap[i]))
        generateModel()
        savepkl()
        loadpkl()
    elif correct == '2'
        break
    plt.close()

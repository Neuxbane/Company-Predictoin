import random
import math
import time
def Sigmoid(x):
	try:
		return 1 / (1 + math.exp(-x))
	except:
		return x*(0<=x<=1)
TransferDerivative = lambda x: x * (1.0 - x)
ReLU = lambda x: x * (x > 0)
rand = lambda a,b: (b-a)*random.random() + a
neux = lambda x: (-1*(x<0.0))+(x>0.0)+(x==0)
class NeuralNetwork:
	def __init__(self,inp,hidlyr,hid,outp,activation='Sigmoid',n=1,minloss=0.4,iterations=-1):
		self.seed = round(time.time())
		random.seed(self.seed)
		counter = time.time()
		self.iterations = iterations
		self.minloss = minloss
		self.testresult = []
		self.n = n
		self.activation = activation
		self.inp = [0 for i in range(inp)]
		self.h = [[0 for i in range(hid)] for i in range(hidlyr)]
		self.outp = [0 for i in range(outp)]
		self.roundoutp = [0 for i in range(outp)]
		self.hw = [[[0 for i in range(hid)] for i in range(hid)] for i in range(hidlyr-1)]
		self.inpw = [[0 for i in range(hid)] for i in range(inp)]
		self.outpw = [[0 for i in range(hid)] for i in range(outp)]
		self.outpb = [0 for i in range(outp)]
		self.hb = [[0 for i in range(hid)] for i in range(hidlyr-1)]
		self.inpb = [0 for i in range(inp)]
		self.random(1)
		#print('\n','Time Spend to Init Network ',time.time()-counter,'\n\n input		=',self.inp,'\n\n','hidden 	= ',self.h,'\n\n','output 	=',self.outp,'\n\n','\n\n weight input to hidden		=',self.inpw,'\n\n','weight hidden 			= ',self.hw,'\n\n','weight hidden to output 	=',self.outpw,'\n\n')
	def calculate(self,a):
		self.h = [[0 for i in range(len(self.h[0]))] for i in range(len(self.h))]
		self.outp = [0 for i in range(len(self.outp))]
		for b in range(len(self.inp)):
			for c in range(len(self.inpw[0])):
				self.inp[b] = a[b]
				self.h[0][c] = (self.h[0][c]+((self.inp[b]*self.inpw[b][c])+self.inpb[b]))
		for b in range(len(self.h)-1):
			for c in range(len(self.hw[0])):
				for d in range(len(self.hw[0][0])):
					self.h[b+1][c] = (self.h[b+1][c]+((self.h[b][c]*self.hw[b][c][d])+self.hb[b][c]))
		for b in range(len(self.outp)):
			for c in range(len(self.outpw[0])):
				self.outp[b] = self.outp[b]+((self.h[len(self.h)-1][c]*self.outpw[b][c])+self.outpb[b])
		exec('self.outp = ['+self.activation+'(x) for x in self.outp]')
		try:
			self.roundoutp = [round(b) for b in self.outp]
		except:
			pass
		self.testresult.append(self.roundoutp)
	def random(self,a):
		for b in range(len(self.h)-1):
			for c in range(len(self.h[0])):
				self.hb[b][c] = (rand(-1*a,a))
				for d in range(len(self.h[0])):
					self.hw[b][c][d] = (rand(-1*a,a))
		for b in range(len(self.inp)):
			self.inpb[b] = (rand(-1*a,a))
			for c in range(len(self.h[0])):
				self.inpw[b][c] = (rand(-1*a,a))
		for b in range(len(self.outp)):
			self.outpb[b] = (rand(-1*a,a))
			for c in range(len(self.h[0])):
				self.outpw[b][c] = (rand(-1*a,a))
	def getloss(self,pola):
		loss = 0
		for a in pola:
			self.calculate(a[0])
			for b in range(len(self.outp)):
				loss += abs((a[1][b]-self.outp[b])*(a[1][b]-self.outp[b]))
		return loss
	def train(self,pola,show=True):
		speed = self.n
		counter = time.time()
		OLoss = 1
		literation = 0
		self.oldloss = 9999999
		self.oldData = [self.inpw,self.hw,self.outpw,self.inpb,self.hb,self.outpb]
		for a in pola:
			self.calculate(a[0])
		while not (((self.getloss(pola)<self.minloss)) or (self.iterations <= literation and self.iterations != -1)):
			try:
				if show:
					print('literation',literation,'loss',self.getloss(pola))
				if 'e+' in str(self.getloss(pola)):
					self.n -= self.n/50
					self.random(1)
					if show:
						print(f'------------------------------------------:Learning Rate set to {self.n}')
				if self.getloss(pola) < self.oldloss:
					self.oldloss = self.getloss(pola)
					self.oldData = [eval(str(self.inpw)+','+str(self.hw)+','+str(self.outpw)+','+str(self.inpb)+','+str(self.hb)+','+str(self.outpb))]
				for b in range(len(self.h)-1):
					for c in range(len(self.h[0])):
						OLoss = self.getloss(pola)
						self.hb[b][c] += 1e-5
						speed = (OLoss-self.getloss(pola))*self.n
						self.hb[b][c] -= 1e-5
						self.hb[b][c] += speed
						for d in range(len(self.h[0])):
							OLoss = self.getloss(pola)
							self.hw[b][c][d] += 1e-5
							speed = (OLoss-self.getloss(pola))*self.n
							self.hw[b][c][d] -= 1e-5
							self.hw[b][c][d] += speed
				for b in range(len(self.inp)):
					OLoss = self.getloss(pola)
					self.inpb[b] += 1e-5
					speed = (OLoss-self.getloss(pola))*self.n
					self.inpb[b] -= 1e-5
					self.inpb[b] += speed
					for c in range(len(self.h[0])):
						OLoss = self.getloss(pola)
						self.inpw[b][c] += 1e-5
						speed = (OLoss-self.getloss(pola))*self.n
						self.inpw[b][c] -= 1e-5
						self.inpw[b][c] += speed
				for b in range(len(self.outp)):
					OLoss = self.getloss(pola)
					self.outpb[b] += 1e-5
					speed = (OLoss-self.getloss(pola))*self.n
					self.outpb[b] -= 1e-5
					self.outpb[b] += speed
					for c in range(len(self.h[0])):
						OLoss = self.getloss(pola)
						self.outpw[b][c] += 1e-5
						speed = (OLoss-self.getloss(pola))*self.n
						self.outpw[b][c] -= 1e-5
						self.outpw[b][c] += speed
				OLoss = self.getloss(pola)
				literation += 1
			except KeyboardInterrupt as e:
				if show:
					print('Litaration End Because KeyboardInterrupt')
					print(e)
				break
		if self.getloss(pola) > self.oldloss:
			self.data_set(self.oldData[0][0],self.oldData[0][1],self.oldData[0][2],self.oldData[0][3],self.oldData[0][4],self.oldData[0][5])
		if show:
			print('DATA-----------------')
			print(str(self.inpw)+','+str(self.hw)+','+str(self.outpw)+','+str(self.inpb)+','+str(self.hb)+','+str(self.outpb))
			print('END------------------')
			print('literation',literation,'loss',self.getloss(pola))
			print('Random seed : ',self.seed)
		print('Done!')
	def test(self,a):
		counter = time.time()
		self.calculate(a)
		return [self.outp,self.roundoutp,time.time()-counter]
	def data_set(self,winput,whidden,woutput,binput,bhidden,boutput):
		self.inpw = winput
		self.outpw = woutput
		self.hw = whidden
		self.inpb = binput
		self.hb = bhidden
		self.outpb = boutput

def wtl(text):
	global letters
	nrange = 20
	b = text+''.join([' ' for x in range(nrange-len(text.split(' ')))])
	return [letters.index(x) for x in b]
def Rwtl(num):
	global letters
	print(num)
	return ''.join([letters[(x<len(letters)-1 and x>0)*x] for x in num])
letters = [' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9',',','*','’','.','?','!','+','-','=','(',')',"'",'"']
#input = ... , hiddenlayer = ... , node per hiddenlayer = ... , output = ...
year = NeuralNetwork(20,1,1,1,activation='',n=0.3,minloss=0.009,iterations=-1)
year.data_set([[1.9661390738096347], [5.098737506627563], [3.09823898085863], [1.7251576251309357], [-1.9685794601288058], [4.2307669247600455], [-2.9825199100487145], [8.850795191805032], [-1.3575019259650414], [-0.045492320563027766], [-0.7307286723688129], [0.19203839046747762], [0.7599408238631773], [-0.7748863444365077], [0.637524143449534], [0.3287584480807453], [0.0889053366868151], [-0.967889838362233], [-0.7580306069118781], [-0.06351081077719001]],[],[[9.360939793510966]],[-0.0861437548882707, 1.6575493455991022, 1.8015223642642104, 1.683091826916706, 0.12665805695591273, 1.2095043403933987, 1.1485855823574374, 1.8103175701159495, 0.5021257340216753, 0.3347981292597075, -0.020314567293285065, -0.13857430003217605, 0.144929165057828, 1.2264590142682177, 1.4102765343095358, 0.2319474262570715, 1.2779221330702724, 0.5538480940147263, 0.11089026443383973, 0.9235430744380011],[],[0.7912502530378376])
year.data_set([[1.1473353508902808], [3.011300524389344], [1.8595377046809227], [1.0160242107395498], [-1.1521205423197218], [2.5345516409147955], [-1.7421882031812168], [5.211940903400617], [-0.822592842371862], [4.089577084603842], [4.410428419005782], [7.217578755407386], [0.8086677046456197], [-0.33042593058784453], [0.562085861718941], [0.6441061630692926], [0.7430783915056054], [0.5606619474192127], [0.27519456416747134], [0.7431483119321041]],[],[[14.371057367848161]],[0.6219666595851143, 0.6836551147175514, 0.42640125493952963, 1.4108115418735756, 2.209319443066819, 1.44980837449298, 0.5396670617472847, 1.3007129590913207, 0.8586231568728094, 1.671686027825069, 2.341499744954497, 0.7595281579434832, 1.3870938482994135, 0.6516672954014312, 1.269864085990213, 1.776944616228726, 0.648387000224575, 1.2338252794033167, 0.4818984130649503, 0.41130094908827836],[],[0.2293742240689513])
year.train([
	[wtl('Apple'),[1976]],
	[wtl('Microsoft'),[1975]],
	[wtl('Google'),[1998]],
	[wtl('Facebook'),[2004]],
	[wtl('Amazon'),[1994]],
	[wtl('GitHub'),[2008]],
	[wtl('Reddit'),[2005]],
	[wtl('Twitter'),[2005]],
	[wtl('Instagram'),[2010]],
	[wtl('Diginet Media'),[2002]]
	],show=1)
while True:
	try:
		c = wtl(input('|<'))
		y = year.test(eval(f"{c}"))
		print('|>',y[1])
	except Exception as e:
		break
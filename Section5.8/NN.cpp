extern "C" {
#include	"NN.h"
}

#include	"JPNeural.h"
using namespace JP;
using namespace JP::Accelerate;

#include	<fstream>
using namespace std;

template	< typename F >	void
Eval(
	Network< F >& n
,	const vMatrix< F >& TestXs
,	const vMatrix< F >& TestAs
) {
	auto wCount = 0;
	auto v = n.Predict( TestXs );
	for ( auto iR = 0; iR < v.nR; iR++ ) {
		auto wMaxIndex = 0;
		for ( auto i = 1; i < v.nC; i++ ) if ( v( iR, i ) > v( iR, wMaxIndex ) ) wMaxIndex = i;
		if ( TestAs( iR, wMaxIndex ) == 1 ) wCount++;
	}
	cerr << "Accuracy: " << ( double( wCount ) / double( TestXs.nR ) ) << endl;
}

template	< typename F >	void
Run(
	const vMatrix< F >& Xs
,	const vMatrix< F >& As
,	const vMatrix< F >& TestXs
,	const vMatrix< F >& TestAs
) {
	Network< F >	wN( Xs.nC );
	wN.NewSigmoidLayer( 50 );
	wN.NewSigmoidLayer( 100 );
	wN.NewSoftmaxLayer( 10 );
	
	ifstream	wI( "/Users/sat/Desktop/NN/sample_weight.bin" );

	switch ( sizeof( F ) ) {
	case 4: {
			double	wBuffer[ 784 * 50 ];
			wI.read( (char*)wBuffer, 784 * 50 * 8 );
			for ( auto i = 0; i < 784 * 50; i++ ) wN.layers[ 0 ]->weight.m[ i ] = wBuffer[ i ];
			wI.read( (char*)wBuffer, 50 * 100 * 8 );
			for ( auto i = 0; i < 50 * 100; i++ ) wN.layers[ 1 ]->weight.m[ i ] = wBuffer[ i ];
			wI.read( (char*)wBuffer, 100 * 10 * 8 );
			for ( auto i = 0; i < 100 * 10; i++ ) wN.layers[ 2 ]->weight.m[ i ] = wBuffer[ i ];

			wI.read( (char*)wBuffer, 50 * 8 );
			for ( auto i = 0; i < 50 ; i++ ) wN.layers[ 0 ]->theta.m[ i ] = wBuffer[ i ];
			wI.read( (char*)wBuffer, 100 * 8 );
			for ( auto i = 0; i < 100; i++ ) wN.layers[ 1 ]->theta.m[ i ] = wBuffer[ i ];
			wI.read( (char*)wBuffer, 10 * 8 );
			for ( auto i = 0; i < 10 ; i++ ) wN.layers[ 1 ]->theta.m[ i ] = wBuffer[ i ];
		}
		break;
	case 8:
		wI.read( (char*)wN.layers[ 0 ]->weight.m, 784 * 50 * 8 );
		wI.read( (char*)wN.layers[ 1 ]->weight.m, 50 * 100 * 8 );
		wI.read( (char*)wN.layers[ 2 ]->weight.m, 100 * 10 * 8 );
		wI.read( (char*)wN.layers[ 0 ]->theta.m, 50 * 8 );
		wI.read( (char*)wN.layers[ 1 ]->theta.m, 100 * 8 );
		wI.read( (char*)wN.layers[ 2 ]->theta.m, 10 * 8 );
		break;
	default:
		assert( false );
	}
	
	Eval( wN, TestXs, TestAs );
//	for ( auto i = 0; i < 1000; i++ ) {
//		wN.Train( Xs, As, 1.0 );
//		Eval( wN, TestXs, TestAs );
//	}
}
int
ReadSwappedUInt32( ifstream& p ) {
	unsigned char	v[ 4 ];
	p.read( (char*)v, 4 );
	return ( v[ 0 ] << 24 ) | ( v[ 1 ] << 16 ) | ( v[ 2 ] << 8 ) | v[ 3 ];
}

template	< typename F >	Matrix< F >
ReadImages( const char* p )  {
	ifstream	wI( p );
	auto		wMagic = ReadSwappedUInt32( wI );	assert( wMagic == 0x00000803 );
	auto		n = ReadSwappedUInt32( wI );
	auto		nPixelsV = ReadSwappedUInt32( wI );
	auto		nPixelsH = ReadSwappedUInt32( wI );
	Matrix< F >	v( n, nPixelsV * nPixelsH );
	for ( auto iR = 0; iR < v.nR; iR++ ) {
		unsigned char	w[ v.nC ];
		wI.read( (char*)w, sizeof( w ) );
		for ( auto iC = 0; iC < v.nC; iC++ ) {
			v( iR, iC ) = w[ iC ] / F( 255 );
		}
	}
	return v;
}

template	< typename F >	Matrix< F >
ReadLabels( string p )  {
	ifstream	wI( p );
	auto		wMagic = ReadSwappedUInt32( wI );	assert( wMagic == 0x00000801 );
	auto		n = ReadSwappedUInt32( wI );
	Matrix< F >	v( n, 10 );
	for ( auto i = 0; i < n; i++ )  v( i, wI.get() ) = 1;
	return v;
}


void
Run(
	const char* pTrainImages
,	const char* pTrainLabels
,	const char* p10kImages
,	const char* p10kLabels
) {
	Run(
		ReadImages< double >( pTrainImages )
	,	ReadLabels< double >( pTrainLabels )
	,	ReadImages< double >( p10kImages )
	,	ReadLabels< double >( p10kLabels )
	);
}


template < typename F > struct
Relu {

	Vector< F >	out;
	
	const Vector< F >&
	forward( const vVector< F >& p ) {
        out = p;
        for ( auto w: out ) if ( w < 0 ) w = 0;
		return out;
	}
	
	const Vector< F >
    backward( const vVector< F >& d ) {
		Vector< F >	v = d;
		for ( auto i = 0; i < out.n; i++ ) if ( out[ i ] == 0 ) v[ i ] = 0;
		return v;
	}
};

template < typename F > struct
Sigmoid {

	vector< F >	out;

	const Vector< F >&
	forward( const vVector< F >& p ) {
		out = Exp( p );
		out = out / ( 1 + out );
 		return out;
	}
	
	const Vector< F >
    backward( const vVector< F >& d ) {
        return d * ( 1 - out ) * out;
	}
};

template < typename F > struct
Affine {
	Matrix< F >	W;
	Vector< F > b;
    Affine( const vMatrix< F >& W, const vVector< F >& b )
    :	W( W )
    ,	b( b ) {
//
//		self.x = None
//		self.original_x_shape = None
//		# 重み・バイアスパラメータの微分
//		self.dW = None
//		self.db = None
	}

	vector< F >	in;
	vector< F >	out;

	const Vector< F >&
	forward( const vVector< F >& p ) {
        in = p;
        out = Dot( p, W ) + b;
        return out;
	}
	
	const Vector< F >
    backward( const vVector< F >& d ) {
		auto v = Dot( W, d );
//		self.dW = Dot( in, d )
//		self.db = SumH( dout )
        return v;
	}
};

template < typename F > struct
SoftmaxWithLoss {

	Vector< F >	t;
	Vector< F >	in;
	Vector< F >	out;
	
	const Vector< F >&
	forward( const vVector< F >& p, const vVector< F >& t ) {
        this->t = t;
        out = softmax( p );
        return cross_entropy_error( out, t );
	}
	
	const Vector< F >
    backward( const vVector< F >& d ) {
		return ( out - t ) / d.n;
	}
};
/*

class Dropout:
//    """
//    http://arxiv.org/abs/1207.0580
//    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
*/
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
*/

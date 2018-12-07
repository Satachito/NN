#include	"JPNeural.h"
using namespace JP;
using namespace JP::Accelerate;

#include	<iostream>
#include	<fstream>

using namespace std;

/*
784 50
50 100
100 10
50
100
10
*/

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
	assert( wI.is_open() );
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

template	< typename F >	void
Main() {
	Run(
		ReadImages< F >( "/Users/sat/Desktop/NN/MNIST/train-images.idx3-ubyte" )
	,	ReadLabels< F >( "/Users/sat/Desktop/NN/MNIST/train-labels.idx1-ubyte" )
	,	ReadImages< F >( "/Users/sat/Desktop/NN/MNIST/t10k-images.idx3-ubyte" )
	,	ReadLabels< F >( "/Users/sat/Desktop/NN/MNIST/t10k-labels.idx1-ubyte" )
	);
}

int
main( int argc, const char * argv[] ) {
	Main< double >();
//	sleep( 10000 );
	return 0;
}

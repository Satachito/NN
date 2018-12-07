//
//  AppDelegate.swift
//  Section5.8
//
//  Created by Satoru Ogura on 2018/12/04.
//  Copyright © 2018年 Satoru Ogura. All rights reserved.
//

import Cocoa

@NSApplicationMain	class
AppDelegate: NSObject, NSApplicationDelegate {
}

class 
Section5_8VC: NSViewController {
	override func viewDidLoad() {
		super.viewDidLoad()
		Run(
			ResourcePath( "train-images", "idx3-ubyte" )!.cString( using: .utf8 )!
		,	ResourcePath( "train-labels", "idx1-ubyte" )!.cString( using: .utf8 )!
		,	ResourcePath( "t10k-images", "idx3-ubyte" )!.cString( using: .utf8 )!
		,	ResourcePath( "t10k-labels", "idx1-ubyte" )!.cString( using: .utf8 )!
		)
	}
}

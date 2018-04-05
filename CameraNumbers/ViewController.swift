//
//  ViewController.swift
//  CameraNumbers
//
//  Created by cl-dev on 2018-04-05.
//  Copyright © 2018 cl-dev. All rights reserved.
//

import UIKit
import AVFoundation
import Vision
import CoreML


class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

  lazy var mlModel: VNCoreMLModel! = {
    do {
      return try VNCoreMLModel(for: MNIST().model)
    } catch {
      fatalError("Could not load MLModel")
    }
  }()

  lazy var cameraFeedView: UIImageView = {
    let imageView = UIImageView()
    imageView.contentMode = .scaleAspectFill
    imageView.clipsToBounds = true
    return imageView
  }()

  lazy var predictionLabel: UILabel = {
    let label = UILabel()
    return label
  }()

  var predictedNumber = "" {
    didSet {
      predictionLabel.text = "Prediction: \(predictedNumber)"
    }
  }

  var captureSession: AVCaptureSession? = nil

  override func viewDidLoad() {
    super.viewDidLoad()

    for v in [predictionLabel, cameraFeedView] as [UIView] {
      v.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview(v)
    }

    cameraFeedView.leftAnchor.constraint(equalTo: view.leftAnchor).isActive = true
    cameraFeedView.rightAnchor.constraint(equalTo: view.rightAnchor, constant: -200).isActive = true
    cameraFeedView.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
    cameraFeedView.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true

    predictionLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor).isActive = true
    predictionLabel.leftAnchor.constraint(equalTo: cameraFeedView.rightAnchor, constant: 8).isActive = true

  }

  override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)

    if AVCaptureDevice.authorizationStatus(for: .video) == .authorized {
      startVideoSession()
    } else {
      AVCaptureDevice.requestAccess(for: .video) { (granted) in
        if granted {
          self.startVideoSession()
        } else {
          // :(
        }
      }
    }
  }

  func startVideoSession() {
    let captureSession = AVCaptureSession()
    // setup inputs
    guard let videoCaptureDevice = AVCaptureDevice.default(for: .video), let videoInput = try? AVCaptureDeviceInput(device: videoCaptureDevice) else {
      return
    }
    captureSession.addInput(videoInput)

    // setup outputs
    let captureOutput = AVCaptureVideoDataOutput()
    captureOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    captureOutput.setSampleBufferDelegate(self, queue: DispatchQueue.main)
    captureSession.addOutput(captureOutput)
    captureSession.startRunning()
    self.captureSession = captureSession
  }

  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let baseImage = ciImageFromSampleBuffer(sampleBuffer) else {
      return
    }
    guard let grayScaleImage = grayScaleImage(baseImage) else {
      return
    }

    guard let negativeGrayScaleImage = negativeImage(grayScaleImage) else {
      return
    }
    cameraFeedView.image = UIImage(ciImage: negativeGrayScaleImage)
    predictNumber(fromImage: negativeGrayScaleImage)
  }

  func ciImageFromSampleBuffer(_ sampleBuffer: CMSampleBuffer) -> CIImage? {
    guard let cvImage = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      return nil
    }
    let ciImage = CIImage(cvPixelBuffer: cvImage)
    return ciImage
  }

  func grayScaleImage(_ inputImage: CIImage) -> CIImage? {
    let grayScaleFilter = CIFilter(name: "CIPhotoEffectNoir")!
    grayScaleFilter.setValue(inputImage, forKey: kCIInputImageKey)
    let outputImage = grayScaleFilter.outputImage
    return outputImage
  }

  func negativeImage(_ inputImage: CIImage) -> CIImage? {
    let negativeFilter = CIFilter(name: "CIColorInvert")!
    negativeFilter.setValue(inputImage, forKey: kCIInputImageKey)
    let outputImage = negativeFilter.outputImage
    return outputImage
  }

  func predictNumber(fromImage image: CIImage) {
    let numberPredictionRequest = VNCoreMLRequest(model: self.mlModel, completionHandler: self.predictionRequestCompletionHandler)
    let numberPredictionHandler = VNImageRequestHandler(ciImage: image, options: [:])
    try? numberPredictionHandler.perform([numberPredictionRequest])
  }

  func predictionRequestCompletionHandler(request: VNRequest, error: Error?) {
    guard error == nil, let results = request.results as? [VNClassificationObservation] else {
      return
    }

    let topResult = results[0]
    DispatchQueue.main.async { [weak self] in
      self?.predictedNumber = topResult.identifier
    }
  }

}

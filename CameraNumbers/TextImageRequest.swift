//
//  TextImageRequest.swift
//  CameraNumbers
//
//  Created by cl-dev on 2018-04-06.
//  Copyright © 2018 cl-dev. All rights reserved.
//

import UIKit
import Vision
import CoreML

typealias ciImageFilter = (CIImage) -> CIImage?

class TextImageRequest {
  private let image: CIImage
  private let filteredImage: CIImage?
  private let mlModel: VNCoreMLModel
  private var completion: (([String]) -> Void)?

  init(image: CIImage, mlModel: VNCoreMLModel) {
    self.image = image
    self.mlModel = mlModel
    filteredImage = TextImageRequest.filterImage(image, filters: [TextImageRequest.grayScaleImage, TextImageRequest.negativeImage])
  }

  static func filterImage(_ image: CIImage, filters: [ciImageFilter]) -> CIImage? {
    var currentImage: CIImage? = image
    for filter in filters {
      currentImage = filter(currentImage!)
      if currentImage == nil {
        return nil
      }
    }
    return currentImage
  }

  static func grayScaleImage(_ inputImage: CIImage) -> CIImage? {
    let grayScaleFilter = CIFilter(name: "CIPhotoEffectNoir")!
    grayScaleFilter.setValue(inputImage, forKey: kCIInputImageKey)
    let outputImage = grayScaleFilter.outputImage
    return outputImage
  }

  static func negativeImage(_ inputImage: CIImage) -> CIImage? {
    let negativeFilter = CIFilter(name: "CIColorInvert")!
    negativeFilter.setValue(inputImage, forKey: kCIInputImageKey)
    let outputImage = negativeFilter.outputImage
    return outputImage
  }

  func predictNumbers(_ completion: @escaping ([String]) -> Void) {
    self.completion = completion
    detectText(in: image)
  }


  private func detectText(in image: CIImage) {
    let request = VNDetectTextRectanglesRequest { (request: VNRequest, error: Error?) in
      guard let observations = request.results as? [VNTextObservation] else {
        return
      }
      // please don't have concurrency issues
      var numbers = [[String]]()

      for (i, observation) in observations.enumerated() {
        numbers.append([])
        for (j, characterBox) in (observation.characterBoxes ?? []).enumerated() {
          numbers[i].append("")
          guard let filteredImage = self.filteredImage else {return}
          let uiImage = UIImage(ciImage: filteredImage)
          let minX = max(0, characterBox.bottomLeft.x)
          let maxX = min(1, characterBox.bottomRight.x)
          let minY = max(0, characterBox.topLeft.y)
          let maxY = min(1, characterBox.bottomLeft.y)
          let originX = minX// * uiImage.size.width
          let originY = minY// * uiImage.size.width
          let width = (maxX - minX)// * uiImage.size.width
          let height = (maxY - minY)// * uiImage.size.width
          guard let croppedImage = self.filteredImage?.cropped(to: CGRect(x: originX, y: originY, width: width, height: height)) else {return}
          let numberPredictionRequest = VNCoreMLRequest(model: self.mlModel) { (request: VNRequest, error: Error?) in
            guard let observations = request.results as? [VNClassificationObservation] else {
              return
            }
            numbers[i][j] = observations.first?.identifier ?? ""
          }
          let numberPredictionHandler = VNImageRequestHandler(ciImage: croppedImage, options: [:])
          try? numberPredictionHandler.perform([numberPredictionRequest])
        }
      }

      self.completion?(numbers.map { return $0.joined() })
    }
    request.reportCharacterBoxes = true
    let imageRequestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    try? imageRequestHandler.perform([request])
  }
}

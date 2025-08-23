"""
Medical Image Processing Component
Handles medical image analysis, identification, and integration with AI learning
"""

import streamlit as st
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import json

# Try to import additional libraries for medical image processing
try:
    from PIL import Image, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)

class MedicalImageProcessor:
    """Process and analyze medical images for AI learning and diagnosis support"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
        self.analysis_cache = {}
        
        # Medical image templates and patterns (simplified)
        self.medical_patterns = {
            'xray': ['chest', 'lung', 'rib', 'bone', 'fracture'],
            'mri': ['brain', 'spine', 'tissue', 'tumor', 'scan'],
            'ct': ['computed', 'tomography', 'slice', 'cross-section'],
            'ultrasound': ['echo', 'sound', 'pregnancy', 'fetal'],
            'microscopy': ['cell', 'tissue', 'bacteria', 'pathology']
        }
    
    def process_medical_image(self, image_file, description: str = "") -> Dict[str, Any]:
        """Process uploaded medical image and extract features for AI learning"""
        try:
            # Convert uploaded file to OpenCV format
            image_array = np.array(Image.open(image_file))
            
            # Convert to BGR for OpenCV processing
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
                
            # Perform medical image analysis
            analysis_results = self._analyze_medical_image(image_bgr, description)
            
            # Extract features for AI learning
            features = self._extract_medical_features(image_bgr)
            
            # Generate learning data
            learning_data = self._generate_learning_data_from_image(
                analysis_results, features, description, image_file.name
            )
            
            return {
                "success": True,
                "analysis": analysis_results,
                "features": features,
                "learning_data": learning_data,
                "image_processed": True
            }
            
        except Exception as e:
            logger.error(f"Error processing medical image: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_processed": False
            }
    
    def _analyze_medical_image(self, image: np.ndarray, description: str) -> Dict[str, Any]:
        """Analyze medical image characteristics"""
        try:
            height, width = image.shape[:2]
            
            # Basic image analysis
            analysis = {
                "dimensions": {"width": width, "height": height},
                "image_type": "grayscale" if len(image.shape) == 2 else "color",
                "estimated_modality": self._detect_medical_modality(description),
                "quality_metrics": self._assess_image_quality(image),
                "anatomical_regions": self._detect_anatomical_regions(description),
                "technical_parameters": self._estimate_technical_parameters(image)
            }
            
            # Advanced analysis if possible
            if analysis["estimated_modality"]:
                analysis["modality_specific"] = self._modality_specific_analysis(
                    image, analysis["estimated_modality"]
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in medical image analysis: {e}")
            return {"error": str(e)}
    
    def _detect_medical_modality(self, description: str) -> Optional[str]:
        """Detect the medical imaging modality from description"""
        description_lower = description.lower()
        
        for modality, keywords in self.medical_patterns.items():
            if any(keyword in description_lower for keyword in keywords):
                return modality
                
        # Check file name patterns
        if 'xray' in description_lower or 'x-ray' in description_lower:
            return 'xray'
        elif 'mri' in description_lower:
            return 'mri'
        elif 'ct' in description_lower or 'cat' in description_lower:
            return 'ct'
        elif 'ultrasound' in description_lower or 'echo' in description_lower:
            return 'ultrasound'
        elif 'microscop' in description_lower or 'histolog' in description_lower:
            return 'microscopy'
            
        return None
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess medical image quality metrics"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate quality metrics
            quality = {
                "brightness": float(np.mean(gray)),
                "contrast": float(np.std(gray)),
                "sharpness": self._calculate_sharpness(gray),
                "noise_level": self._estimate_noise_level(gray),
                "quality_score": 0.0
            }
            
            # Calculate overall quality score
            quality["quality_score"] = self._calculate_quality_score(quality)
            
            return quality
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return {"error": str(e)}
    
    def _calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            return float(laplacian.var())
        except:
            return 0.0
    
    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in the image"""
        try:
            # Use median filter to estimate noise
            median_filtered = cv2.medianBlur(gray_image, 5)
            noise = np.abs(gray_image.astype(float) - median_filtered.astype(float))
            return float(np.mean(noise))
        except:
            return 0.0
    
    def _calculate_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics"""
        try:
            # Normalize metrics (simplified scoring)
            brightness_score = min(quality_metrics["brightness"] / 128.0, 1.0)
            contrast_score = min(quality_metrics["contrast"] / 64.0, 1.0)
            sharpness_score = min(quality_metrics["sharpness"] / 1000.0, 1.0)
            noise_score = max(0, 1.0 - quality_metrics["noise_level"] / 50.0)
            
            # Weighted average
            overall_score = (
                brightness_score * 0.2 + 
                contrast_score * 0.3 + 
                sharpness_score * 0.3 + 
                noise_score * 0.2
            )
            
            return float(overall_score)
            
        except:
            return 0.5
    
    def _detect_anatomical_regions(self, description: str) -> List[str]:
        """Detect anatomical regions mentioned in description"""
        anatomical_terms = {
            'chest': ['chest', 'thorax', 'lung', 'heart', 'rib'],
            'head': ['head', 'brain', 'skull', 'cranium', 'cerebral'],
            'abdomen': ['abdomen', 'stomach', 'liver', 'kidney', 'intestine'],
            'spine': ['spine', 'vertebra', 'spinal', 'back'],
            'extremities': ['arm', 'leg', 'hand', 'foot', 'joint', 'bone'],
            'pelvis': ['pelvis', 'hip', 'pelvic']
        }
        
        detected_regions = []
        description_lower = description.lower()
        
        for region, terms in anatomical_terms.items():
            if any(term in description_lower for term in terms):
                detected_regions.append(region)
                
        return detected_regions
    
    def _estimate_technical_parameters(self, image: np.ndarray) -> Dict[str, Any]:
        """Estimate technical parameters of the medical image"""
        try:
            height, width = image.shape[:2]
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            parameters = {
                "resolution": f"{width}x{height}",
                "bit_depth": "8-bit" if gray.dtype == np.uint8 else "16-bit",
                "pixel_spacing": "unknown",  # Would need DICOM metadata
                "window_level": float(np.mean(gray)),
                "window_width": float(np.std(gray) * 2),
                "estimated_body_part": "unknown"
            }
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error estimating technical parameters: {e}")
            return {"error": str(e)}
    
    def _modality_specific_analysis(self, image: np.ndarray, modality: str) -> Dict[str, Any]:
        """Perform modality-specific analysis"""
        try:
            if modality == "xray":
                return self._analyze_xray(image)
            elif modality == "mri":
                return self._analyze_mri(image)
            elif modality == "ct":
                return self._analyze_ct(image)
            elif modality == "ultrasound":
                return self._analyze_ultrasound(image)
            elif modality == "microscopy":
                return self._analyze_microscopy(image)
            else:
                return {"analysis": "general_medical_image"}
                
        except Exception as e:
            logger.error(f"Error in modality-specific analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_xray(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze X-ray specific features"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect high-contrast areas (bones)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bone_area = np.sum(binary > 0) / (gray.shape[0] * gray.shape[1])
        
        return {
            "modality": "X-ray",
            "bone_visibility": "high" if bone_area > 0.3 else "medium" if bone_area > 0.1 else "low",
            "contrast_level": "high" if np.std(gray) > 50 else "medium",
            "suggested_findings": "Examine for fractures, pneumonia, or structural abnormalities"
        }
    
    def _analyze_mri(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze MRI specific features"""
        return {
            "modality": "MRI", 
            "tissue_contrast": "excellent",
            "suggested_findings": "Analyze soft tissue structures, tumors, or lesions"
        }
    
    def _analyze_ct(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze CT scan specific features"""
        return {
            "modality": "CT",
            "cross_sectional": True,
            "suggested_findings": "Examine for masses, hemorrhages, or structural changes"
        }
    
    def _analyze_ultrasound(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze ultrasound specific features"""
        return {
            "modality": "Ultrasound",
            "real_time_imaging": True,
            "suggested_findings": "Check for fluid collections, organ structure, or fetal development"
        }
    
    def _analyze_microscopy(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze microscopy specific features"""
        return {
            "modality": "Microscopy",
            "cellular_level": True,
            "suggested_findings": "Examine cellular structures, pathogens, or tissue abnormalities"
        }
    
    def _extract_medical_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features that can be used for AI learning"""
        try:
            # Convert to grayscale for feature extraction
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Extract various features
            features = {
                "histogram": cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist(),
                "edges": self._extract_edge_features(gray),
                "texture": self._extract_texture_features(gray),
                "shape_features": self._extract_shape_features(gray),
                "intensity_stats": {
                    "mean": float(np.mean(gray)),
                    "std": float(np.std(gray)),
                    "min": float(np.min(gray)),
                    "max": float(np.max(gray))
                }
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting medical features: {e}")
            return {"error": str(e)}
    
    def _extract_edge_features(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Extract edge-related features"""
        try:
            edges = cv2.Canny(gray_image, 50, 150)
            edge_density = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
            
            return {
                "edge_density": edge_density,
                "edge_strength": "high" if edge_density > 0.1 else "medium" if edge_density > 0.05 else "low"
            }
        except:
            return {"edge_density": 0.0, "edge_strength": "unknown"}
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Extract texture-related features"""
        try:
            # Simple texture measures
            local_std = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            
            return {
                "texture_complexity": float(local_std),
                "texture_level": "high" if local_std > 1000 else "medium" if local_std > 500 else "low"
            }
        except:
            return {"texture_complexity": 0.0, "texture_level": "unknown"}
    
    def _extract_shape_features(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Extract shape-related features"""
        try:
            # Find contours for shape analysis
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                return {
                    "num_objects": len(contours),
                    "largest_area": float(area),
                    "largest_perimeter": float(perimeter),
                    "circularity": float(4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
                }
            else:
                return {"num_objects": 0}
                
        except:
            return {"num_objects": 0}
    
    def _generate_learning_data_from_image(self, analysis: Dict, features: Dict, 
                                         description: str, filename: str) -> Dict[str, Any]:
        """Generate learning data that can be fed to the AI model"""
        try:
            # Create comprehensive learning content
            learning_content = f"""
Medical Image Analysis Report:

Image: {filename}
Description: {description}

Technical Analysis:
- Modality: {analysis.get('estimated_modality', 'Unknown')}
- Dimensions: {analysis.get('dimensions', {})}
- Quality Score: {analysis.get('quality_metrics', {}).get('quality_score', 'Unknown')}

Clinical Observations:
- Anatomical Regions: {', '.join(analysis.get('anatomical_regions', []))}
- Image Quality: {analysis.get('quality_metrics', {}).get('contrast', 'Unknown')} contrast level
- Technical Parameters: {analysis.get('technical_parameters', {})}

AI Learning Points:
- This medical image demonstrates {analysis.get('estimated_modality', 'medical imaging')} characteristics
- Quality assessment shows {analysis.get('quality_metrics', {}).get('quality_score', 0.5):.2f} score
- Features extracted for pattern recognition and diagnosis support
- Clinical correlation recommended for comprehensive assessment

Medical Recommendations:
- Professional radiologist interpretation advised
- Consider clinical context and patient history
- Follow appropriate medical imaging protocols
"""

            # Create question-answer pairs for AI training
            qa_pairs = [
                {
                    "question": f"What type of medical image is this: {filename}?",
                    "answer": f"This is a {analysis.get('estimated_modality', 'medical')} image showing {', '.join(analysis.get('anatomical_regions', ['anatomical structures']))}."
                },
                {
                    "question": f"What is the quality assessment of this medical image?",
                    "answer": f"The image quality score is {analysis.get('quality_metrics', {}).get('quality_score', 0.5):.2f} with {analysis.get('quality_metrics', {}).get('contrast', 'moderate')} contrast."
                },
                {
                    "question": f"What should be considered when interpreting this medical image?",
                    "answer": f"Consider the {analysis.get('estimated_modality', 'imaging')} modality characteristics, image quality, and clinical correlation with patient symptoms and history."
                }
            ]
            
            return {
                "learning_content": learning_content,
                "qa_pairs": qa_pairs,
                "image_metadata": {
                    "filename": filename,
                    "modality": analysis.get('estimated_modality'),
                    "quality_score": analysis.get('quality_metrics', {}).get('quality_score', 0.5),
                    "processed_timestamp": time.time()
                },
                "knowledge_points": len(qa_pairs)
            }
            
        except Exception as e:
            logger.error(f"Error generating learning data: {e}")
            return {"error": str(e)}
    
    def render_image_processing_interface(self):
        """Render the medical image processing interface in Streamlit"""
        st.subheader("🖼️ Medical Image Analysis & Learning")
        st.write("Upload medical images for AI analysis and learning enhancement")
        
        # File uploader
        uploaded_images = st.file_uploader(
            "Upload Medical Images",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Supported formats: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_images:
            for image_file in uploaded_images:
                with st.expander(f"🔬 Process: {image_file.name}", expanded=True):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Display the image
                        image = Image.open(image_file)
                        st.image(image, caption=f"Medical Image: {image_file.name}", use_column_width=True)
                    
                    with col2:
                        # Image description input
                        description = st.text_area(
                            "Medical Image Description",
                            placeholder="e.g., 'Chest X-ray showing lungs and ribcage' or 'MRI brain scan axial view'",
                            key=f"desc_{image_file.name}"
                        )
                        
                        # Process button
                        if st.button(f"🧠 Analyze & Learn from Image", key=f"process_{image_file.name}"):
                            with st.spinner(f"Analyzing {image_file.name}..."):
                                # Process the image
                                result = self.process_medical_image(image_file, description)
                                
                                if result["success"]:
                                    st.success("✅ Medical image analyzed successfully!")
                                    
                                    # Display analysis results
                                    analysis = result["analysis"]
                                    
                                    # Basic info
                                    st.write("**🔍 Analysis Results:**")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Modality", analysis.get("estimated_modality", "Unknown"))
                                    with col2:
                                        quality_score = analysis.get("quality_metrics", {}).get("quality_score", 0)
                                        st.metric("Quality Score", f"{quality_score:.2f}")
                                    with col3:
                                        regions = analysis.get("anatomical_regions", [])
                                        st.metric("Regions Found", len(regions))
                                    
                                    # Detailed analysis
                                    if st.checkbox(f"Show Detailed Analysis for {image_file.name}", key=f"detail_{image_file.name}"):
                                        st.json(analysis)
                                    
                                    # Learning integration
                                    learning_data = result["learning_data"]
                                    if learning_data and "knowledge_points" in learning_data:
                                        st.info(f"🎓 Generated {learning_data['knowledge_points']} knowledge points for AI learning")
                                        
                                        # Add to intelligent model
                                        if st.button(f"📚 Add to AI Knowledge Base", key=f"learn_{image_file.name}"):
                                            try:
                                                from components.intelligent_model import intelligent_model
                                                learn_result = intelligent_model.add_document_and_learn(
                                                    learning_data["learning_content"],
                                                    f"Medical Image: {image_file.name}"
                                                )
                                                if "error" not in learn_result:
                                                    st.success("🧠 Image knowledge added to AI brain!")
                                                    st.metric("Knowledge Points Added", learn_result["knowledge_points_extracted"])
                                                else:
                                                    st.error(f"Learning failed: {learn_result['error']}")
                                            except Exception as e:
                                                st.error(f"Error adding to knowledge base: {e}")
                                else:
                                    st.error(f"❌ Image analysis failed: {result.get('error', 'Unknown error')}")
        else:
            st.info("📤 Upload medical images to begin AI-powered analysis and learning")
            
            # Example information
            with st.expander("ℹ️ Supported Medical Image Types"):
                st.write("""
                **Supported Medical Imaging Modalities:**
                - 🫁 **X-Ray**: Chest, bone, dental radiographs
                - 🧠 **MRI**: Brain, spine, soft tissue scans  
                - 🔬 **CT**: Cross-sectional imaging, angiograms
                - 👶 **Ultrasound**: Pregnancy, cardiac, abdominal
                - 🦠 **Microscopy**: Histology, pathology, cellular imaging
                
                **AI Learning Benefits:**
                - Extract medical features automatically
                - Generate training data for improved diagnosis
                - Build knowledge base of medical patterns
                - Enhance AI's understanding of medical imaging
                """)

# Global instance
medical_image_processor = MedicalImageProcessor()
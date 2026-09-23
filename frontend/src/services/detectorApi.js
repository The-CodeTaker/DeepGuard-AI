import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000/api";

const IMAGE_API_URL = `${API_BASE_URL}/predict/`;
const VIDEO_API_URL = `${API_BASE_URL}/predict-video/`;

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("image", file);

  const response = await axios.post(IMAGE_API_URL, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
}

export async function analyzeVideo(file) {
  const formData = new FormData();
  formData.append("video", file);

  const response = await axios.post(VIDEO_API_URL, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
}

export async function analyzeMedia(file, mediaType) {
  if (mediaType === "video") {
    return analyzeVideo(file);
  }

  return analyzeImage(file);
}
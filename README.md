# Signature Extractor API

A Python Flask API that extracts handwritten signatures from PDFs or ID images and returns them as base64-encoded PNGs.

## Endpoint

`POST /extract-signature`

**Form field**: `file` (PDF or image)

**Response**:
```json
{
  "signature_base64": "<Base64String>"
}

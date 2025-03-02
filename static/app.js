const fileInput = document.getElementById('fileInput');
const fileTypeSelect = document.getElementById('file_type');
const anonymizeBtn = document.getElementById('anonymizeBtn');
const status = document.getElementById('status');
const downloadBtn = document.getElementById('downloadBtn');
const backendUrl = 'http://localhost:8000';

let uploadedFileName = '';

fileInput.addEventListener('change', () => {
    anonymizeBtn.disabled = !fileInput.files.length;
    status.textContent = fileInput.files.length ? 'File selected. Click "Anonymize" to process.' : '';
});

async function downloadFile(event) {
   try {
       const fileName = event.target.value;
       const response = await fetch(`${backendUrl}/download/${fileName}`)

       if (!response.ok) {
           throw new Error(response.detail);
       }

       const blobFile = await response.blob();
       const ulr = window.URL.createObjectURL(blobFile);

       const link = document.createElement("a");
       link.href = ulr;
       link.download = event.target.value;
       document.body.appendChild(link);

       link.click();

       document.body.removeChild(link);
       window.URL.revokeObjectURL(ulr);

       status.textContent = "Successfully downloaded"

   } catch (error) {
       console.log("Download is failed:" , error)
       status.textContent = `Download is failed: ${error}`
   }
}

anonymizeBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        status.textContent = 'Please select a file first!';
        return;
    }

    anonymizeBtn.disabled = true;
    status.textContent = 'Uploading file...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const uploadResponse = await fetch(`${backendUrl}/upload/`, {
            method: 'POST',
            body: formData
        });
        const uploadData = await uploadResponse.json();
        if (!uploadResponse.ok) throw new Error(uploadData.detail || 'Upload failed');
        uploadedFileName = uploadData.file_name;
        status.textContent = 'File uploaded. Anonymizing...';

        const fileType = fileTypeSelect.value;

        const anonymizeResponse = await fetch(`${backendUrl}/anonymize/${uploadedFileName}?file_type=${fileType}`);
        const anonymizeData = await anonymizeResponse.json();
        if (!anonymizeResponse.ok) throw new Error(anonymizeData.detail || 'Anonymization failed');
        status.textContent = 'Anonymization complete!';
        const anonymizedFileName = anonymizeData.file_name;

        downloadBtn.value = anonymizedFileName.split("/").at(-1);
        downloadBtn.addEventListener("click", downloadFile);
        downloadBtn.disabled = false;
    } catch (error) {
        status.textContent = `Error: ${error.message}`;
    } finally {
        anonymizeBtn.disabled = false;
    }
});
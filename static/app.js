const fileInput = document.getElementById('fileInput');
const fileTypeSelect = document.getElementById('file_type'); // Added file type select
const anonymizeBtn = document.getElementById('anonymizeBtn');
const status = document.getElementById('status');
const downloadLink = document.getElementById('downloadLink');
const backendUrl = 'http://localhost:8000';

let uploadedFileName = '';

fileInput.addEventListener('change', () => {
    anonymizeBtn.disabled = !fileInput.files.length;
    status.textContent = fileInput.files.length ? 'File selected. Click "Anonymize" to process.' : '';
    downloadLink.style.display = 'none';
});

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

        debugger;
        const anonymizeResponse = await fetch(`${backendUrl}/anonymize/${uploadedFileName}?file_type=${fileType}`);
        const anonymizeData = await anonymizeResponse.json();
        if (!anonymizeResponse.ok) throw new Error(anonymizeData.detail || 'Anonymization failed');
        const anonymizedFileName = anonymizeData.file_name;
        status.textContent = 'Anonymization complete!';

        downloadLink.href = `${backendUrl}/download/${anonymizedFileName}`;
        downloadLink.download = anonymizedFileName;
        downloadLink.style.display = 'block';
        downloadLink.textContent = `Download ${anonymizedFileName}`;
    } catch (error) {
        status.textContent = `Error: ${error.message}`;
    } finally {
        anonymizeBtn.disabled = false;
    }
});
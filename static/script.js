function clearForm() {

    // Clear the textarea
    document.querySelector("textarea").value = "";

    // Reload the home page to clear all results
    window.location.href = "/";
}
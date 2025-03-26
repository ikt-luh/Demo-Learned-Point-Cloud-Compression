   // JavaScript to handle button selection
   document.addEventListener('DOMContentLoaded', () => {
    const buttons = document.querySelectorAll('.quality-button');

    buttons.forEach(button => {
        button.addEventListener('click', async (event) => {
            event.preventDefault();

            // Remove 'active' class from all buttons
            buttons.forEach(btn => btn.classList.remove('active'));

            // Add 'active' class to the clicked button
            button.classList.add('active');

            // Get the selected quality value
            const selectedQuality = button.dataset.quality;
            console.log('Selected Quality:', selectedQuality);

            // Send the selected quality to the server via a POST request
            try {
                const response = await fetch("/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    body: `fixed_quality=${selectedQuality}`
                });

                if (!response.ok) {
                    console.error("Failed to update quality:", response.statusText);
                } else {
                    console.log("Quality updated successfully");
                }
            } catch (error) {
                console.error("Error updating quality:", error);
            }
        });
    });
});
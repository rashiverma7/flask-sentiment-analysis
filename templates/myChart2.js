function renderChart(data, labels) {
    var ctx = document.getElementById("myChart").getContext('2d');
    var myChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: 'This week',
                data: data,
                backgroundColor: [
                        "#FF6384",
                        "#63FF84",
                        "#ff9100",
                        "#84FF63",
                        "#8463FF",
                        "#6384FF",
                        "#ff9100",
                    ],
                //borderColor: 'rgba(255,165,0, 1)',
                //backgroundColor: 'rgba(255,165,0, 0.2)',
            }]
        },
    });
}

$("#sub1").click(
    function () {
        data = [20000, 14000, 12000, 15000, 18000, 19000, 22000];
        labels =  ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"];
        backgroundColor: [
                "#FF6384",
                "#63FF84",
                "#84FF63",
                "#8463FF",
                "#6384FF"
            ],
        renderChart(data, labels);
    }
);

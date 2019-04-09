function openPage(){

  let url = 'http://127.0.0.1:5000/review-evaluation'
  fetch(url)
  .then(response => response.json())
  .then(data => 

    console.log(JSON.stringify(data.reviews[0])))
  .catch(function(err){
    if (err) throw err;
  });
}


  
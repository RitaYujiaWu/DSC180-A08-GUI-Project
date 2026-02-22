// DSC180 A08 — Website (minimal JS for nav highlight / optional dropdown)

(function () {
  'use strict';

  // Optional: highlight current section in nav on scroll
  var sections = document.querySelectorAll('.section[id]');
  var navLinks = document.querySelectorAll('.nav-inner a[href^="#"]');

  function updateNavHighlight() {
    var scrollY = window.scrollY || window.pageYOffset;
    var innerHeight = window.innerHeight * 0.4;
    var current = null;
    sections.forEach(function (section) {
      var top = section.offsetTop;
      var height = section.offsetHeight;
      if (scrollY >= top - innerHeight && scrollY < top + height - innerHeight) {
        current = section.getAttribute('id');
      }
    });
    navLinks.forEach(function (link) {
      var href = link.getAttribute('href');
      if (href === '#' + current) {
        link.setAttribute('aria-current', 'location');
        link.classList.add('current');
      } else {
        link.removeAttribute('aria-current');
        link.classList.remove('current');
      }
    });
  }

  if (navLinks.length && sections.length) {
    window.addEventListener('scroll', function () {
      requestAnimationFrame(updateNavHighlight);
    });
    updateNavHighlight();
  }

  // Report link: replace #report-link with your actual PDF URL when you have it
  var reportLink = document.getElementById('report-link');
  if (reportLink && reportLink.getAttribute('href') === '#report-link') {
    reportLink.setAttribute('href', '#');
    reportLink.addEventListener('click', function (e) {
      e.preventDefault();
      // When you have a PDF URL, set it here or in HTML: href="path/to/report.pdf"
      console.log('Add your report PDF URL to the Report link in index.html');
    });
  }
})();

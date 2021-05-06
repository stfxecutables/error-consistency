

<!DOCTYPE html>
<html class="writer-html5" lang="en" >
<head>
  <meta charset="utf-8" />
  
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  
  <title>error-consistency documentation</title>
  

  
  <link rel="stylesheet" href="_static/css/theme.css" type="text/css" />
  <link rel="stylesheet" href="_static/pygments.css" type="text/css" />
  <link rel="stylesheet" href="_static/overrides.css" type="text/css" />

  
  

  
  

  

  
  <!--[if lt IE 9]>
    <script src="_static/js/html5shiv.min.js"></script>
  <![endif]-->
  
    
      <script type="text/javascript" id="documentation_options" data-url_root="./" src="_static/documentation_options.js"></script>
        <script src="_static/jquery.js"></script>
        <script src="_static/underscore.js"></script>
        <script src="_static/doctools.js"></script>
        <script src="_static/style_force.js"></script>
    
    <script type="text/javascript" src="_static/js/theme.js"></script>

    
    <link rel="index" title="Index" href="genindex.html" />
    <link rel="search" title="Search" href="search.html" /> 
</head>

<body class="wy-body-for-nav">

   
  <div class="wy-grid-for-nav">
    
    <nav data-toggle="wy-nav-shift" class="wy-nav-side">
      <div class="wy-side-scroll">
        <div class="wy-side-nav-search"  style="background: white" >
          

          
            <a href="#" class="icon icon-home"> error-consistency
          

          
          </a>

          
            
            
          

          

          
        </div>

        
        <div class="wy-menu wy-menu-vertical" data-spy="affix" role="navigation" aria-label="main navigation">
          
            
            
            
              <!-- Local TOC -->
              <div class="local-toc"><p class="caption"><span class="caption-text">Contents:</span></p>
<ul>
<li class="toctree-l1"><a class="reference internal" href="index.html#document-quickstart">Quickstart</a></li>
<li class="toctree-l1"><a class="reference internal" href="index.html#document-coreapi">Core API</a></li>
<li class="toctree-l1"><a class="reference internal" href="index.html#document-consistency">Error Consistency Classes</a></li>
<li class="toctree-l1"><a class="reference internal" href="index.html#document-functional">Functional Tools</a></li>
<li class="toctree-l1"><a class="reference internal" href="index.html#document-containers">Containers</a></li>
<li class="toctree-l1"><a class="reference internal" href="index.html#document-model">Model Tools</a></li>
<li class="toctree-l1"><a class="reference internal" href="index.html#document-theory">Error Consistency Theory</a><ul>
<li class="toctree-l2"><a class="reference internal" href="index.html#choice-of-groupings">Choice of Groupings</a></li>
<li class="toctree-l2"><a class="reference internal" href="index.html#why-p-2">Why p=2</a></li>
</ul>
</li>
</ul>
</div>
            
          
        </div>
        
      </div>
    </nav>

    <section data-toggle="wy-nav-shift" class="wy-nav-content-wrap">

      
      <nav class="wy-nav-top" aria-label="top navigation">
        
          <i data-toggle="wy-nav-top" class="fa fa-bars"></i>
          <a href="#">error-consistency</a>
        
      </nav>


      <div class="wy-nav-content">
        
        <div class="rst-content">
        
          

















<div role="navigation" aria-label="breadcrumbs navigation">

  <ul class="wy-breadcrumbs">
    
      <li><a href="#" class="icon icon-home"></a> &raquo;</li>
        
      <li>error-consistency  documentation</li>
    
    
      <li class="wy-breadcrumbs-aside">
        
          
        
      </li>
    
  </ul>

  
  <hr/>
</div>
          <div role="main" class="document" itemscope="itemscope" itemtype="http://schema.org/Article">
           <div itemprop="articleBody">
            
  <div class="section" id="documentation">
<h1>Documentation<a class="headerlink" href="#documentation" title="Permalink to this headline">¶</a></h1>
<div class="toctree-wrapper compound">
<span id="document-quickstart"></span><div class="section" id="quickstart">
<h2>Quickstart<a class="headerlink" href="#quickstart" title="Permalink to this headline">¶</a></h2>
<p>To fit a standard scikit-learn classifier, or any classifier with either <code class="docutils literal notranslate"><span class="pre">.fit</span></code> or <code class="docutils literal notranslate"><span class="pre">.train</span></code> and
<code class="docutils literal notranslate"><span class="pre">.predict</span></code> or <code class="docutils literal notranslate"><span class="pre">.test</span></code> methods, using <a class="reference internal" href="index.html#error_consistency.consistency.ErrorConsistencyKFoldHoldout" title="error_consistency.consistency.ErrorConsistencyKFoldHoldout"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.consistency.ErrorConsistencyKFoldHoldout</span></code></a></p>
<div class="highlight-python3 notranslate"><div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">error_consistency.consistency</span> <span class="kn">import</span> <span class="n">ErrorConsistencyKFoldHoldout</span> <span class="k">as</span> <span class="n">ErrorConsistency</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span> <span class="k">as</span> <span class="n">KNN</span>

<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">])</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">])</span>

<span class="n">knn_args</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">errcon</span> <span class="o">=</span> <span class="n">ErrorConsistency</span><span class="p">(</span><span class="n">KNN</span><span class="p">,</span> <span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">n_splits</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">model_args</span><span class="o">=</span><span class="n">knn_args</span><span class="p">)</span>
<span class="n">results</span> <span class="o">=</span> <span class="n">errcon</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span>
   <span class="n">x_test</span><span class="p">,</span>
   <span class="n">y_test</span><span class="p">,</span>
   <span class="n">repetitions</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span>
   <span class="n">show_progress</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
   <span class="n">parallel_reps</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
   <span class="n">loo_parallel</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span>
   <span class="n">turbo</span><span class="o">=</span><span class="kc">True</span>
<span class="p">)</span>

<span class="c1"># 10 reps * 5 splits = 50 errors sets</span>
<span class="nb">print</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">results</span><span class="o">.</span><span class="n">consistencies</span><span class="p">))</span>  <span class="c1"># 50*(50-1) // 2</span>
</pre></div>
</div>
<p>To evaluate the error consistency of a set of predictions on a test set with <a class="reference internal" href="index.html#error_consistency.functional.error_consistencies" title="error_consistency.functional.error_consistencies"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.functional.error_consistencies</span></code></a>:</p>
<div class="highlight-python3 notranslate"><div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">error_consistency.functional</span> <span class="kn">import</span> <span class="n">error_consistencies</span>
<span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span> <span class="k">as</span> <span class="n">KNN</span>

<span class="c1"># random training set</span>
<span class="n">N_TRAIN</span> <span class="o">=</span> <span class="mi">10</span>
<span class="n">x_trains</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span> <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N_TRAIN</span><span class="p">)]</span>
<span class="n">y_trains</span> <span class="o">=</span> <span class="p">[</span><span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">])</span> <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N_TRAIN</span><span class="p">)]</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">50</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">50</span><span class="p">])</span>
<span class="n">y_preds</span> <span class="o">=</span> <span class="p">[</span><span class="n">KNN</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span><span class="p">,</span> <span class="n">y</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">x_trains</span><span class="p">,</span> <span class="n">y_trains</span><span class="p">)]</span>

<span class="c1"># only grab consistencies and matrix</span>
<span class="n">consistencies</span><span class="p">,</span> <span class="n">matrix</span> <span class="o">=</span> <span class="n">error_consistencies</span><span class="p">(</span><span class="n">y_preds</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)[</span><span class="mi">0</span><span class="p">:</span><span class="mi">2</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="n">matrix</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>  <span class="c1"># (10, 10)</span>
</pre></div>
</div>
</div>
<span id="document-coreapi"></span><div class="section" id="core-api">
<h2>Core API<a class="headerlink" href="#core-api" title="Permalink to this headline">¶</a></h2>
<table class="longtable docutils align-default">
<colgroup>
<col style="width: 10%" />
<col style="width: 90%" />
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="index.html#error_consistency.consistency.ErrorConsistencyKFoldHoldout" title="error_consistency.consistency.ErrorConsistencyKFoldHoldout"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.consistency.ErrorConsistencyKFoldHoldout</span></code></a></p></td>
<td><p>Compute error consistencies for a classifier.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="index.html#error_consistency.functional.error_consistencies" title="error_consistency.functional.error_consistencies"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.functional.error_consistencies</span></code></a></p></td>
<td><p>Get the error consistency for a list of predictions.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="index.html#error_consistency.containers.ConsistencyResults" title="error_consistency.containers.ConsistencyResults"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.containers.ConsistencyResults</span></code></a></p></td>
<td><p>Holds results from evaluating error consistency.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="index.html#error_consistency.model.Model" title="error_consistency.model.Model"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.model.Model</span></code></a></p></td>
<td><p>Helper class for making a unified interface to different model types.</p></td>
</tr>
</tbody>
</table>
</div>
<span id="document-consistency"></span><div class="section" id="module-error_consistency.consistency">
<span id="error-consistency-classes"></span><h2>Error Consistency Classes<a class="headerlink" href="#module-error_consistency.consistency" title="Permalink to this headline">¶</a></h2>
<dl class="py class">
<dt id="error_consistency.consistency.ErrorConsistency">
<em class="property"><span class="pre">class</span> </em><code class="sig-prename descclassname"><span class="pre">error_consistency.consistency.</span></code><code class="sig-name descname"><span class="pre">ErrorConsistency</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">model</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">x</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">n_splits</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">model_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args_x_y</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args_x</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">stratify</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">x_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">empty_unions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">onehot_y</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.consistency.ErrorConsistency" title="Permalink to this definition">¶</a></dt>
<dd><p>Compute the error consistency of a classifier.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>model</strong> (<em>Intersection</em><em>[</em><em>Callable</em><em>, </em><em>Type</em><em>]</em>) – <p>A <em>class</em> where instances are classifiers that implement:</p>
<ol class="arabic simple">
<li><p>A <code class="docutils literal notranslate"><span class="pre">.fit</span></code> or <code class="docutils literal notranslate"><span class="pre">.train</span></code> method that:</p>
<ol class="arabic simple">
<li><p>accepts predictors and targets, plus <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit_args</span></code>, and</p></li>
<li><p>updates the state of <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> when calling <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code></p></li>
</ol>
</li>
<li><p>A <code class="docutils literal notranslate"><span class="pre">.predict</span></code> or <code class="docutils literal notranslate"><span class="pre">.test</span></code> method, that:</p>
<ol class="arabic simple">
<li><p>accepts testing samples, plus <code class="docutils literal notranslate"><span class="pre">predict_args</span></code>, and</p></li>
<li><p>requires having called <code class="docutils literal notranslate"><span class="pre">.fit</span></code> previously, and</p></li>
<li><p>returns <em>only</em> the predictions as a single ArrayLike (e.g. NumPy array, List, pandas
DataFrame or Series)</p></li>
</ol>
</li>
</ol>
<p>E.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">error_consistency</span> <span class="kn">import</span> <span class="n">ErrorConsistency</span>
<span class="kn">from</span> <span class="nn">sklearn.cluster</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span> <span class="k">as</span> <span class="n">KNN</span>

<span class="n">knn_args</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">errcon</span> <span class="o">=</span> <span class="n">ErrorConsistency</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="n">KNN</span><span class="p">,</span> <span class="n">model_args</span><span class="o">=</span><span class="n">knn_args</span><span class="p">)</span>

<span class="c1"># KNN is appropriate here because we could write e.g.</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">])</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">])</span>

<span class="n">KNN</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>  <span class="c1"># updates the state, no need to use a returned value</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">KNN</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>  <span class="c1"># returns a single object</span>
</pre></div>
</div>
</p></li>
<li><p><strong>x</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – <p>ArrayLike object containing predictor samples. Must be in a format that is consumable with
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code> for arguments <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">model_args</span></code>. If using external
validation (e.g. passing <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> into <a class="reference internal" href="#error_consistency.consistency.ErrorConsistency.evaluate" title="error_consistency.consistency.ErrorConsistency.evaluate"><code class="xref py py-obj docutils literal notranslate"><span class="pre">ErrorConsistency.evaluate</span></code></a>), you must ensure <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code>
does not contain <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code>, that is, this argument functions as if it is <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_train</span></code>.</p>
<p>Otherwise, if using internal validation, splitting of x into validation subsets will be
along the first axis (axis 0), that is, the first axis is assumed to be the sample
dimension. If your fit method requires a different sample dimension, you can specify this
in <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_sample_dim</span></code>.</p>
</p></li>
<li><p><strong>y</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – <p>ArrayLike object containing targets. Must be in a format that is consumable with
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code> for arguments <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">model_args</span></code>. If using external
validation (e.g. passing <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> into <a class="reference internal" href="#error_consistency.consistency.ErrorConsistency.evaluate" title="error_consistency.consistency.ErrorConsistency.evaluate"><code class="xref py py-obj docutils literal notranslate"><span class="pre">ErrorConsistency.evaluate</span></code></a>), you must ensure <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code>
does not contain <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code>, that is, this argument functions as if it is <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_train</span></code>.</p>
<p>Otherwise, if using internal validation, splitting of y into validation subsets will be
along the first axis (axis 0), that is, the first axis is assumed to be the sample
dimension. If your fit method requires a different sample dimension (e.g. y is a one-hot
encoded array), you can specify this in <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_sample_dim</span></code>.</p>
</p></li>
<li><p><strong>n_splits</strong> (<em>int = 5</em>) – How many folds to use for validating error consistency. Only relevant</p></li>
<li><p><strong>model_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time to construct a fresh instance of the model (see
above). Note that the data x and y must NOT be included here.</p></li>
<li><p><strong>fit_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> methods
internally (see notes for <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> above). Note that the data x and y must NOT be included
here.</p></li>
<li><p><strong>fit_args_x_y</strong> (<em>Optional</em><em>[</em><em>Tuple</em><em>[</em><em>str</em><em>, </em><em>str</em><em>]</em><em>] </em><em>= None</em>) – <p>Name of the arguments which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> and target <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> are passed to. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code>.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> method of the instance of
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> as its second, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code>.</p>
<p>If a tuple of strings (x_name, y_name), then a dict will be constructed internally by
splatting, e.g.</p>
<blockquote>
<div><p>args_dict = {<a href="#id1"><span class="problematic" id="id2">**</span></a>{x_name: x_train, y_name: y_train}, <a href="#id3"><span class="problematic" id="id4">**</span></a>model_args}
model.fit(<a href="#id5"><span class="problematic" id="id6">**</span></a>args_dict)</p>
</div></blockquote>
</p></li>
<li><p><strong>predict_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> methods
internally (see notes for <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> above). Note that the data x must NOT be included here.</p></li>
<li><p><strong>predict_args_x</strong> (<em>Optional</em><em>[</em><em>str</em><em>] </em><em>= None</em>) – <p>Name of the argument which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> is passed to during evaluation. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> calls.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> method of the instance
of <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.predict(x,</span> <span class="pre">**predict_args)</span></code>.</p>
<p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict_args_x</span></code> is a string, then a dict will be constructed internally with this
string, e.g.</p>
<blockquote>
<div><p>args_dict = {<a href="#id7"><span class="problematic" id="id8">**</span></a>{predict_args_x: x_train}, <a href="#id9"><span class="problematic" id="id10">**</span></a>model_args}
model.predict(<a href="#id11"><span class="problematic" id="id12">**</span></a>args_dict)</p>
</div></blockquote>
</p></li>
<li><p><strong>stratify</strong> (<em>bool = False</em>) – If True, use sklearn.model_selection.StratifiedKFold during internal k-fold. Otherwise, use
sklearn.model_selection.KFold.</p></li>
<li><p><strong>x_sample_dim</strong> (<em>int = 0</em>) – The axis or dimension along which samples are indexed. Needed for splitting x into
partitions for k-fold.</p></li>
<li><p><strong>y_sample_dim</strong> (<em>int = 0</em>) – The axis or dimension along which samples are indexed. Needed for splitting y into
partitions for k-fold only if the target is e.g. one-hot encoded or dummy-coded.</p></li>
<li><p><strong>empty_unions</strong> (<em>UnionHandling = 0</em>) – <p>When computing the pairwise consistency or leave-one-out consistency on small or
simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
prediction errors are made). In this case the intersection over union is 0/0, which is
undefined.</p>
<ul>
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">0</span></code> (default), the consistency for that collection of error sets is set to zero.</p></li>
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">1</span></code>, the consistency for that collection of error sets is set to one.</p></li>
<li><p>If “nan”, the consistency for that collection of error sets is set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nan</span></code>.</p></li>
<li><p>If “drop”, the <code class="xref py py-obj docutils literal notranslate"><span class="pre">consistencies</span></code> array will not include results for that collection,
but the consistency matrix will include <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nans</span></code>.</p></li>
<li><p>If “error”, an empty union will cause a <code class="xref py py-obj docutils literal notranslate"><span class="pre">ZeroDivisionError</span></code>.</p></li>
<li><p>If “warn”, an empty union will print a warning (probably a lot).</p></li>
</ul>
</p></li>
<li><p><strong>onehot_y</strong> (<em>bool = True</em>) – Only relevant for two-dimensional <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code>. Set to True if <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> is a one-hot array with samples
indexed by <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_sample_dim</span></code>. Set to False if <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> is dummy-coded.</p></li>
</ul>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>Conceptually, for each repetition, there are two steps to computing a k-fold error consistency
with holdout set:</p>
<blockquote>
<div><ol class="arabic simple">
<li><p>evaluation on standard k-fold (“validation” or “folding”)</p></li>
<li><p>evaluation on holdout set (outside of k-fold) (“testing”)</p></li>
</ol>
</div></blockquote>
<p>There are a lot of overlapping terms and concepts here, so with analogy to deep learning, we
shall refer to step (1) as <em>validation</em> or <em>val</em> and step (2) as <em>testing</em> or <em>test</em>. This will
help keep variable names and function arguments sane and clear. We refer to the <em>entire</em> process
of validation + testing as <em>evaluation</em>. Thus the .evaluate() method with have both validation
and testing steps, in this terminology.</p>
<p>Since validation is very much just standard k-fold, we also thus refer to validation steps as
<em>fold</em> steps. So for example validation or fold scores are the k accuracies on the non-training
partitions of each k-fold repetition (k*repetitions total), but test scores are the
<code class="xref py py-obj docutils literal notranslate"><span class="pre">repititions</span></code> accuracies on the heldout test set.</p>
<p>The good thing is that standard k-fold is standard k-fold no matter how we implement
error-consistency (e.g. with holdout, Monte-Carlo style subsetting, etc). We just have train and
(fold) test indices, and do the usual fit calls and etc. So this can be abstracted to the base
error consistency class.</p>
<dl class="field-list simple">
</dl>
<dl class="py method">
<dt id="error_consistency.consistency.ErrorConsistency.evaluate">
<code class="sig-name descname"><span class="pre">evaluate</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">repetitions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">5</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">x_test</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_test</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_errors</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_predictions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_preds</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_models</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">empty_unions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">show_progress</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">parallel_reps</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">loo_parallel</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">turbo</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">seed</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.consistency.ErrorConsistency.evaluate" title="Permalink to this definition">¶</a></dt>
<dd><p>Evaluate the error consistency of the classifier.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>repetitions</strong> (<em>int = 5</em>) – How many times to repeat the k-fold process. Yields <code class="xref py py-obj docutils literal notranslate"><span class="pre">k*repetitions</span></code> error consistencies
if both <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are provided, and <code class="xref py py-obj docutils literal notranslate"><span class="pre">repetitions*(repititions</span> <span class="pre">-</span> <span class="pre">1)/2</span></code>
consistencies otherwise. Note that if both <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are not provided, then
setting repetitions to 1 will raise an error, since this results in insufficient arrays
to compare errors.</p></li>
<li><p><strong>x_test</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – ArrayLike object containing holdout predictor samples that the model will never be
trained or fitted on. Must be have a format identical to that of <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> passed into
constructor (see above).</p></li>
<li><p><strong>y_test</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – ArrayLike object containing holdout target values that the model will never be trained
or fitted on. Must be have a format identical to that of <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> passed into constructor
(see above).</p></li>
<li><p><strong>save_test_accs</strong> (<em>bool = True</em>) – <p>If True (default) also compute accuracy scores and save them in the returned
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.scores</span></code>. If False, skip this step.</p>
<p>Note: when <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are provided, test accuracies are over these values.
When not provided, test accuracies are over the entire set <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> passed into the
<a class="reference internal" href="#error_consistency.consistency.ErrorConsistency" title="error_consistency.consistency.ErrorConsistency"><code class="xref py py-obj docutils literal notranslate"><span class="pre">ErrorConsistency</span></code></a> constructor, but constructed from each fold (e.g. if there are <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span></code>
splits, the predictions on the k disjoint folds are joined together to get one total
set of predictions for that repetition).</p>
</p></li>
<li><p><strong>save_test_errors</strong> (<em>bool = False</em>) – <p>If True, save a list of the boolean error arrays (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred</span> <span class="pre">!=</span> <span class="pre">y_test</span></code>) for all
repetitions. If False (default), the return value <code class="xref py py-obj docutils literal notranslate"><span class="pre">results</span></code> will have
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_errors</span></code> be <code class="xref py py-obj docutils literal notranslate"><span class="pre">None</span></code>.</p>
<p>Note: when <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are provided, errors are on <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code>.
When not provided, test accuracies are over the entire set <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> passed into the
<a class="reference internal" href="#error_consistency.consistency.ErrorConsistency" title="error_consistency.consistency.ErrorConsistency"><code class="xref py py-obj docutils literal notranslate"><span class="pre">ErrorConsistency</span></code></a> constructor, but constructed from each fold (e.g. if there are <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span></code>
splits, the predictions on the k disjoint folds are joined together to get one total
set of predictions for that repetition).</p>
</p></li>
<li><p><strong>save_test_predictions</strong> (<em>bool = False</em>) – <p>If True, save an array of the predictions <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code> for all repetitions in
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_predictions</span></code>. Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1. If False
(default), <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_predictions</span></code> will be <code class="xref py py-obj docutils literal notranslate"><span class="pre">None</span></code>.</p>
<p>Note: when <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are provided, predictions are for <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code>.
When not provided, predictions are for the entire set <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> passed into the
<a class="reference internal" href="#error_consistency.consistency.ErrorConsistency" title="error_consistency.consistency.ErrorConsistency"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.consistency.ErrorConsistency</span></code></a> constructor, but constructed from the
models trained on each disjoint fold (e.g. if there are <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span></code> splits, the predictions on
the <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span></code> disjoint folds are joined together to get one total set of predictions for that
repetition). That is, the predictions are the combined results of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span></code> different models.</p>
</p></li>
<li><p><strong>save_fold_accs</strong> (<em>bool = False</em>) – <p>If True, save a list of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k)</span></code> of the predictions on the <em>fold</em> test
sets for all repetitions. This list will be available in <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_accs</span></code>. If False,
do not save these values.</p>
<p>Note: when <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are provided, and <code class="xref py py-obj docutils literal notranslate"><span class="pre">save_fold_accs=False</span></code> and
<code class="xref py py-obj docutils literal notranslate"><span class="pre">save_fold_preds=False</span></code>, then the entire prediction and accuracy evaluation on each
k-fold will be skipped, potentially saving significant compute time, depending on the
model and size of the dataset. However, when using an internal validation method
(<code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are not provided) this prediction step still must be executed.</p>
</p></li>
<li><p><strong>save_fold_preds</strong> (<em>bool = False</em>) – If True, save a list with shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k,</span> <span class="pre">n_samples)</span></code> of the predictions on
the <em>fold</em> test set for all repetitions. This list will be abalable in
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_predictions</span></code>. If False, do not save these values. See Notes above for
extra details on this behaviour.</p></li>
<li><p><strong>save_fold_models</strong> (<em>bool = False</em>) – <p>If True, <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_models</span></code> is a nested list of size (repetitions, k) where
each entry (r, i) is the <em>fitted</em> model on repetition <code class="xref py py-obj docutils literal notranslate"><span class="pre">r</span></code> fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>.</p>
<p>Note: During parallelization, new models are constructed each time using the passed in
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> class and the model arguments.Parallelization pickles these models and the
associated data, and then the actual models are fit in each separate process. When
there is no parallelization, the procedure is still similar, in that separate models
are created for every repetition. Thus, you have to be careful about memory when using
<code class="xref py py-obj docutils literal notranslate"><span class="pre">save_fold_models</span></code> and a large number of repetions. The <code class="xref py py-obj docutils literal notranslate"><span class="pre">error-consistency</span></code> library
wraps all <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> classes passed in into a <code class="xref py py-obj docutils literal notranslate"><span class="pre">Model</span></code> class which is used internally to
unify interfacing across various libraries. This <code class="xref py py-obj docutils literal notranslate"><span class="pre">Model</span></code> class is very tiny, and is not
a concern for memory, but if the wrapped model is large, you may have memory problems.
E.g. KNN and other memory-based methods which may have an option <code class="xref py py-obj docutils literal notranslate"><span class="pre">save_x_y</span></code> or the like
could lead to problems when using <code class="xref py py-obj docutils literal notranslate"><span class="pre">save_fold_models=True</span></code>.</p>
</p></li>
<li><p><strong>seed</strong> (<em>int = None</em>) – Seed for reproducible results.</p></li>
<li><p><strong>empty_unions</strong> (<em>error_consistency.functional.UnionHandling</em>) – </p></li>
<li><p><strong>show_progress</strong> (<em>bool</em>) – </p></li>
<li><p><strong>parallel_reps</strong> (<em>bool</em>) – </p></li>
<li><p><strong>loo_parallel</strong> (<em>bool</em>) – </p></li>
<li><p><strong>turbo</strong> (<em>bool</em>) – </p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p><strong>results</strong> – An <a class="reference internal" href="index.html#error_consistency.containers.ConsistencyResults" title="error_consistency.containers.ConsistencyResults"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.containers.ConsistencyResults</span></code></a> object.</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p><a class="reference internal" href="index.html#error_consistency.containers.ConsistencyResults" title="error_consistency.containers.ConsistencyResults">ConsistencyResults</a></p>
</dd>
</dl>
</dd></dl>

</dd></dl>

<dl class="py class">
<dt id="error_consistency.consistency.ErrorConsistencyKFoldHoldout">
<em class="property"><span class="pre">class</span> </em><code class="sig-prename descclassname"><span class="pre">error_consistency.consistency.</span></code><code class="sig-name descname"><span class="pre">ErrorConsistencyKFoldHoldout</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">model</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">x</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">n_splits</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">model_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args_x_y</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args_x</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">stratify</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">x_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">empty_unions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.consistency.ErrorConsistencyKFoldHoldout" title="Permalink to this definition">¶</a></dt>
<dd><p>Compute error consistencies for a classifier.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><p><strong>model</strong> (<em>Intersection</em><em>[</em><em>Callable</em><em>, </em><em>Type</em><em>]</em>) – <p>A <em>class</em> where instances are classifiers that implement:</p>
<ol class="arabic simple">
<li><p>A <code class="docutils literal notranslate"><span class="pre">.fit</span></code> or <code class="docutils literal notranslate"><span class="pre">.train</span></code> method that:</p>
<ul class="simple">
<li><p>accepts predictors and targets, plus <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit_args</span></code>, and</p></li>
<li><p>updates the state of <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> when calling <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code></p></li>
</ul>
</li>
<li><p>A <code class="docutils literal notranslate"><span class="pre">.predict</span></code> or <code class="docutils literal notranslate"><span class="pre">.test</span></code> method, that:</p>
<ul class="simple">
<li><p>accepts testing samples, plus <code class="docutils literal notranslate"><span class="pre">predict_args</span></code>, and</p></li>
<li><p>requires having called <code class="docutils literal notranslate"><span class="pre">.fit</span></code> previously, and</p></li>
<li><p>returns <em>only</em> the predictions as a single ArrayLike (e.g. NumPy array, List, pandas
DataFrame or Series)</p></li>
</ul>
</li>
</ol>
<span class="target" id="valid-model-example"></span><p>E.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">error_consistency</span> <span class="kn">import</span> <span class="n">ErrorConsistency</span>
<span class="kn">from</span> <span class="nn">sklearn.cluster</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span> <span class="k">as</span> <span class="n">KNN</span>

<span class="n">knn_args</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">errcon</span> <span class="o">=</span> <span class="n">ErrorConsistency</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="n">KNN</span><span class="p">,</span> <span class="n">model_args</span><span class="o">=</span><span class="n">knn_args</span><span class="p">)</span>

<span class="c1"># KNN is appropriate here because we could write e.g.</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">])</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">])</span>

<span class="n">KNN</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>  <span class="c1"># updates the state, no need to use a returned value</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">KNN</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>  <span class="c1"># returns a single object</span>
</pre></div>
</div>
</p>
</dd>
</dl>
<dl>
<dt>x: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]</dt><dd><p>ArrayLike object containing predictor samples. Must be in a format that is consumable with
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code> for arguments <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">model_args</span></code>. By default,
splitting of x into cross-validation subsets will be along the first axis (axis 0), that is,
the first axis is assumed to be the sample dimension. If your fit method requires a
different sample dimension, you can specify this in <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_sample_dim</span></code>.</p>
</dd>
<dt>y: Union[List, pandas.DataFrame, pandas.Series, numpy.ndarray]</dt><dd><p>ArrayLike object containing targets. Must be in a format that is consumable with
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code> for arguments <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">model_args</span></code>. By default,
splitting of y into cross-validation subsets will be along the first axis (axis 0), that is,
the first axis is assumed to be the sample dimension. If your fit method requires a
different sample dimension (e.g. y is a one-hot encoded array), you can specify this
in <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_sample_dim</span></code>.</p>
</dd>
<dt>n_splits: int = 5</dt><dd><p>How many folds to use, and thus models to generate, per repetition.</p>
</dd>
<dt>model_args: Optional[Dict[str, Any]]</dt><dd><p>Any arguments that are required each time to construct a fresh instance of the model (see
the <a href="#id14"><span class="problematic" id="id15">`valid model example`_</span></a> above). Note that the data x and y must NOT be included here.</p>
</dd>
<dt>fit_args: Optional[Dict[str, Any]]</dt><dd><p>Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> methods
internally (see the <a href="#id16"><span class="problematic" id="id17">`valid model example`_</span></a> above). Note that the data x and y must NOT be
included here.</p>
</dd>
<dt>fit_args_x_y: Optional[Tuple[str, str]] = None</dt><dd><p>Name of the arguments which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> and target <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> are passed to. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code>. For example, a function may have the
signature:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">f</span><span class="p">(</span><span class="n">predictor</span><span class="p">:</span> <span class="n">ndarray</span><span class="p">,</span> <span class="n">target</span><span class="p">:</span> <span class="n">ndarray</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Any</span>
</pre></div>
</div>
<p>To allow our internal <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_train</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> splits to be passed to the right arguments,
we thus need to know these names.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> method of the instance of
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> as its second, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code>.</p>
<p>If a tuple of strings (x_name, y_name), then a dict will be constructed internally by
splatting, e.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">args_dict</span> <span class="o">=</span> <span class="p">{</span><span class="o">**</span><span class="p">{</span><span class="n">x_name</span><span class="p">:</span> <span class="n">x_train</span><span class="p">,</span> <span class="n">y_name</span><span class="p">:</span> <span class="n">y_train</span><span class="p">},</span> <span class="o">**</span><span class="n">model_args</span><span class="p">}</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="o">**</span><span class="n">args_dict</span><span class="p">)</span>
</pre></div>
</div>
<p>Alternately, see the documentation for <a class="reference internal" href="index.html#error_consistency.model.Model" title="error_consistency.model.Model"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.model.Model</span></code></a> for how to subclass
your own function here if you require more fine-grained control of how arguments are passed
into the fit and predict calls.</p>
</dd>
<dt>predict_args: Optional[Dict[str, Any]]</dt><dd><p>Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> methods
internally (see the <a href="#id18"><span class="problematic" id="id19">`valid model example`_</span></a> above). Note that the data x must NOT be included
here.</p>
</dd>
<dt>predict_args_x: Optional[str] = None</dt><dd><p>Name of the argument which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> is passed to during evaluation. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> calls.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> method of the instance
of <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.predict(x,</span> <span class="pre">**predict_args)</span></code>.</p>
<p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict_args_x</span></code> is a string, then a dict will be constructed internally with this
string, e.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">args_dict</span> <span class="o">=</span> <span class="p">{</span><span class="o">**</span><span class="p">{</span><span class="n">predict_args_x</span><span class="p">:</span> <span class="n">x_train</span><span class="p">},</span> <span class="o">**</span><span class="n">model_args</span><span class="p">}</span>
<span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="o">**</span><span class="n">args_dict</span><span class="p">)</span>
</pre></div>
</div>
</dd>
<dt>stratify: bool = False</dt><dd><p>If True, use sklearn.model_selection.StratifiedKFold during internal k-fold. Otherwise, use
sklearn.model_selection.KFold.</p>
</dd>
<dt>x_sample_dim: int = 0</dt><dd><p>The axis or dimension along which samples are indexed. Needed for splitting x into
partitions for k-fold.</p>
</dd>
<dt>y_sample_dim: int = 0</dt><dd><p>The axis or dimension along which samples are indexed. Needed for splitting y into
partitions for k-fold only if the target is e.g. one-hot encoded or dummy-coded.</p>
</dd>
<dt>empty_unions: UnionHandling = 0</dt><dd><p>When computing the pairwise consistency or leave-one-out consistency on small or
simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
prediction errors are made). In this case the intersection over union is 0/0, which is
undefined.</p>
<ul class="simple">
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">0</span></code> (default), the consistency for that collection of error sets is set to zero.</p></li>
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">1</span></code>, the consistency for that collection of error sets is set to one.</p></li>
<li><p>If “nan”, the consistency for that collection of error sets is set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nan</span></code>.</p></li>
<li><p>If “drop”, the <code class="xref py py-obj docutils literal notranslate"><span class="pre">consistencies</span></code> array will not include results for that collection,
but the consistency matrix will include <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nans</span></code>.</p></li>
<li><p>If “error”, an empty union will cause a <code class="xref py py-obj docutils literal notranslate"><span class="pre">ZeroDivisionError</span></code>.</p></li>
<li><p>If “warn”, an empty union will print a warning (probably a lot).</p></li>
</ul>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>Conceptually, for each repetition, there are two steps to computing a k-fold error consistency
with holdout set:</p>
<blockquote>
<div><ol class="arabic simple">
<li><p>evaluation on standard k-fold (“validation” or “folding”)</p></li>
<li><p>evaluation on holdout set (outside of k-fold) (“testing”)</p></li>
</ol>
</div></blockquote>
<p>There are a lot of overlapping terms and concepts here, so with analogy to deep learning, we
shall refer to step (1) as <em>validation</em> or <em>val</em> and step (2) as <em>testing</em> or <em>test</em>. This will
help keep variable names and function arguments sane and clear. We refer to the <em>entire</em> process
of validation + testing as <em>evaluation</em>. Thus the .evaluate() method with have both validation
and testing steps, in this terminology.</p>
<p>Since validation is very much just standard k-fold, we also thus refer to validation steps as
<em>fold</em> steps. So for example validation or fold scores are the k accuracies on the non-training
partitions of each k-fold repetition (k*repetitions total), but test scores are the
<code class="xref py py-obj docutils literal notranslate"><span class="pre">repititions</span></code> accuracies on the heldout test set.</p>
<dl class="py method">
<dt id="error_consistency.consistency.ErrorConsistencyKFoldHoldout.evaluate">
<code class="sig-name descname"><span class="pre">evaluate</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">x_test</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_test</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">repetitions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">5</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_errors</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_predictions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_preds</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_models</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">empty_unions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">show_progress</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">parallel_reps</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">loo_parallel</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">turbo</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">seed</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.consistency.ErrorConsistencyKFoldHoldout.evaluate" title="Permalink to this definition">¶</a></dt>
<dd><p>Evaluate the error consistency of the classifier.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>x_test</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – ArrayLike object containing holdout predictor samples that the model will never be
trained or fitted on. Must be have a format identical to that of <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> passed into
constructor (see above).</p></li>
<li><p><strong>y_test</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – ArrayLike object containing holdout target values that the model will never be trained
or fitted on. Must be have a format identical to that of <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> passed into constructor
(see above).</p></li>
<li><p><strong>repetitions</strong> (<em>int = 5</em>) – How many times to repeat the k-fold process. Yields <code class="xref py py-obj docutils literal notranslate"><span class="pre">k*repetitions</span></code> error consistencies
if both <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are proided, and <code class="xref py py-obj docutils literal notranslate"><span class="pre">repetitions*(repititions</span> <span class="pre">-</span> <span class="pre">1)/2</span></code>
consistencies otherwise. Note that if both <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_test</span></code> are not provided, then
setting repetitions to 1 will raise an error, since this results in insufficient arrays
to compare errors.</p></li>
<li><p><strong>save_test_accs</strong> (<em>bool = True</em>) – If True (default) also compute accuracy scores for each fold on <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and save them
in <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.scores</span></code>. If False, skip this step. Setting to <code class="xref py py-obj docutils literal notranslate"><span class="pre">False</span></code> is useful when
prediction is expensive and/or you only care about evaulating the error consistency.</p></li>
<li><p><strong>save_test_errors</strong> (<em>bool = False</em>) – If True, save a list of the boolean error arrays (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_i</span> <span class="pre">!=</span> <span class="pre">y_test</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for
all repetitions in <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_errors</span></code>. Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1.
If False (default), <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_errors</span></code> will be <code class="xref py py-obj docutils literal notranslate"><span class="pre">None</span></code>.</p></li>
<li><p><strong>save_test_predictions</strong> (<em>bool = False</em>) – If True, save an array of the predictions <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code> for all repetitions in
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_predictions</span></code>. Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1. If False
(default), <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_predictions</span></code> will be <code class="xref py py-obj docutils literal notranslate"><span class="pre">None</span></code>.</p></li>
<li><p><strong>save_fold_accs</strong> (<em>bool = False</em>) – If True, save an array of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k)</span></code> of the predictions on the <em>fold</em> test
set (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_fold_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for all repetitions in <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_accs</span></code>.</p></li>
<li><p><strong>save_fold_preds</strong> (<em>bool = False</em>) – If True, save a NumPy array of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k,</span> <span class="pre">n_samples)</span></code> of the predictions on
the <em>fold</em> test set (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_fold_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for all repetitions in
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_predictions</span></code>.</p></li>
<li><p><strong>save_fold_models</strong> (<em>bool = False</em>) – If True, <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_models</span></code> is a NumPy object array of size (repetitions, k) where
each entry (r, i) is the fitted model on repetition <code class="xref py py-obj docutils literal notranslate"><span class="pre">r</span></code> fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>.</p></li>
<li><p><strong>seed</strong> (<em>int = None</em>) – Seed for reproducible results</p></li>
<li><p><strong>empty_unions</strong> (<em>error_consistency.functional.UnionHandling</em>) – </p></li>
<li><p><strong>show_progress</strong> (<em>bool</em>) – </p></li>
<li><p><strong>parallel_reps</strong> (<em>Union</em><em>[</em><em>bool</em><em>, </em><em>int</em><em>]</em>) – </p></li>
<li><p><strong>loo_parallel</strong> (<em>Union</em><em>[</em><em>bool</em><em>, </em><em>int</em><em>]</em>) – </p></li>
<li><p><strong>turbo</strong> (<em>bool</em>) – </p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p><strong>results</strong> – An <a class="reference internal" href="index.html#error_consistency.containers.ConsistencyResults" title="error_consistency.containers.ConsistencyResults"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.containers.ConsistencyResults</span></code></a> object.</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p><a class="reference internal" href="index.html#error_consistency.containers.ConsistencyResults" title="error_consistency.containers.ConsistencyResults">ConsistencyResults</a></p>
</dd>
</dl>
</dd></dl>

</dd></dl>

<dl class="py class">
<dt id="error_consistency.consistency.ErrorConsistencyKFoldInternal">
<em class="property"><span class="pre">class</span> </em><code class="sig-prename descclassname"><span class="pre">error_consistency.consistency.</span></code><code class="sig-name descname"><span class="pre">ErrorConsistencyKFoldInternal</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">model</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">x</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">n_splits</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">model_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args_x_y</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args_x</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">stratify</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">x_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">empty_unions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.consistency.ErrorConsistencyKFoldInternal" title="Permalink to this definition">¶</a></dt>
<dd><p>Compute error consistencies for a classifier.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>model</strong> (<em>Intersection</em><em>[</em><em>Callable</em><em>, </em><em>Type</em><em>]</em>) – <p>A <em>class</em> where instances are classifiers that implement:</p>
<ol class="arabic simple">
<li><p>A <code class="docutils literal notranslate"><span class="pre">.fit</span></code> or <code class="docutils literal notranslate"><span class="pre">.train</span></code> method that:</p>
<ul>
<li><p>accepts predictors and targets, plus <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit_args</span></code>, and</p></li>
<li><p>updates the state of <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> when calling <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code></p></li>
</ul>
</li>
<li><p>A <code class="docutils literal notranslate"><span class="pre">.predict</span></code> or <code class="docutils literal notranslate"><span class="pre">.test</span></code> method, that:</p>
<ul>
<li><p>accepts testing samples, plus <code class="docutils literal notranslate"><span class="pre">predict_args</span></code>, and</p></li>
<li><p>requires having called <code class="docutils literal notranslate"><span class="pre">.fit</span></code> previously, and</p></li>
<li><p>returns <em>only</em> the predictions as a single ArrayLike (e.g. NumPy array, List, pandas
DataFrame or Series)</p></li>
</ul>
</li>
</ol>
<span class="target" id="id13"></span><p>E.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">error_consistency</span> <span class="kn">import</span> <span class="n">ErrorConsistency</span>
<span class="kn">from</span> <span class="nn">sklearn.cluster</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span> <span class="k">as</span> <span class="n">KNN</span>

<span class="n">knn_args</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">errcon</span> <span class="o">=</span> <span class="n">ErrorConsistency</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="n">KNN</span><span class="p">,</span> <span class="n">model_args</span><span class="o">=</span><span class="n">knn_args</span><span class="p">)</span>

<span class="c1"># KNN is appropriate here because we could write e.g.</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">100</span><span class="p">])</span>
<span class="n">x_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">uniform</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">,</span> <span class="mi">5</span><span class="p">])</span>
<span class="n">y_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="p">[</span><span class="mi">20</span><span class="p">])</span>

<span class="n">KNN</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>  <span class="c1"># updates the state, no need to use a returned value</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">KNN</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">x_test</span><span class="p">)</span>  <span class="c1"># returns a single object</span>
</pre></div>
</div>
</p></li>
<li><p><strong>x</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – ArrayLike object containing predictor samples. Must be in a format that is consumable with
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code> for arguments <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">model_args</span></code>. By default,
splitting of x into cross-validation subsets will be along the first axis (axis 0), that is,
the first axis is assumed to be the sample dimension. If your fit method requires a
different sample dimension, you can specify this in <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_sample_dim</span></code>.</p></li>
<li><p><strong>y</strong> (<em>Union</em><em>[</em><em>List</em><em>, </em><em>pandas.DataFrame</em><em>, </em><em>pandas.Series</em><em>, </em><em>numpy.ndarray</em><em>]</em>) – ArrayLike object containing targets. Must be in a format that is consumable with
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code> for arguments <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">model_args</span></code>. By default,
splitting of y into cross-validation subsets will be along the first axis (axis 0), that is,
the first axis is assumed to be the sample dimension. If your fit method requires a
different sample dimension (e.g. y is a one-hot encoded array), you can specify this
in <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_sample_dim</span></code>.</p></li>
<li><p><strong>n_splits</strong> (<em>int = 5</em>) – How many folds to use, and thus models to generate, per repetition.</p></li>
<li><p><strong>model_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time to construct a fresh instance of the model (see
the <a href="#id20"><span class="problematic" id="id21">`valid model example`_</span></a> above). Note that the data x and y must NOT be included here.</p></li>
<li><p><strong>fit_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> methods
internally (see the <a href="#id22"><span class="problematic" id="id23">`valid model example`_</span></a> above). Note that the data x and y must NOT be
included here.</p></li>
<li><p><strong>fit_args_x_y</strong> (<em>Optional</em><em>[</em><em>Tuple</em><em>[</em><em>str</em><em>, </em><em>str</em><em>]</em><em>] </em><em>= None</em>) – <p>Name of the arguments which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> and target <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> are passed to. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code>. For example, a function may have the
signature:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">f</span><span class="p">(</span><span class="n">predictor</span><span class="p">:</span> <span class="n">ndarray</span><span class="p">,</span> <span class="n">target</span><span class="p">:</span> <span class="n">ndarray</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Any</span>
</pre></div>
</div>
<p>To allow our internal <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_train</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> splits to be passed to the right arguments,
we thus need to know these names.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> method of the instance of
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> as its second, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code>.</p>
<p>If a tuple of strings (x_name, y_name), then a dict will be constructed internally by
splatting, e.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">args_dict</span> <span class="o">=</span> <span class="p">{</span><span class="o">**</span><span class="p">{</span><span class="n">x_name</span><span class="p">:</span> <span class="n">x_train</span><span class="p">,</span> <span class="n">y_name</span><span class="p">:</span> <span class="n">y_train</span><span class="p">},</span> <span class="o">**</span><span class="n">model_args</span><span class="p">}</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="o">**</span><span class="n">args_dict</span><span class="p">)</span>
</pre></div>
</div>
<p>Alternately, see the documentation for <a class="reference internal" href="index.html#error_consistency.model.Model" title="error_consistency.model.Model"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.model.Model</span></code></a> for how to subclass
your own function here if you require more fine-grained control of how arguments are passed
into the fit and predict calls.</p>
</p></li>
<li><p><strong>predict_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> methods
internally (see the <a href="#id24"><span class="problematic" id="id25">`valid model example`_</span></a> above). Note that the data x must NOT be included
here.</p></li>
<li><p><strong>predict_args_x</strong> (<em>Optional</em><em>[</em><em>str</em><em>] </em><em>= None</em>) – <p>Name of the argument which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> is passed to during evaluation. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> calls.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> method of the instance
of <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.predict(x,</span> <span class="pre">**predict_args)</span></code>.</p>
<p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict_args_x</span></code> is a string, then a dict will be constructed internally with this
string, e.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">args_dict</span> <span class="o">=</span> <span class="p">{</span><span class="o">**</span><span class="p">{</span><span class="n">predict_args_x</span><span class="p">:</span> <span class="n">x_train</span><span class="p">},</span> <span class="o">**</span><span class="n">model_args</span><span class="p">}</span>
<span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="o">**</span><span class="n">args_dict</span><span class="p">)</span>
</pre></div>
</div>
</p></li>
<li><p><strong>stratify</strong> (<em>bool = False</em>) – If True, use sklearn.model_selection.StratifiedKFold during internal k-fold. Otherwise, use
sklearn.model_selection.KFold.</p></li>
<li><p><strong>x_sample_dim</strong> (<em>int = 0</em>) – The axis or dimension along which samples are indexed. Needed for splitting x into
partitions for k-fold.</p></li>
<li><p><strong>y_sample_dim</strong> (<em>int = 0</em>) – The axis or dimension along which samples are indexed. Needed for splitting y into
partitions for k-fold only if the target is e.g. one-hot encoded or dummy-coded.</p></li>
<li><p><strong>empty_unions</strong> (<em>UnionHandling = 0</em>) – <p>When computing the pairwise consistency or leave-one-out consistency on small or
simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
prediction errors are made). In this case the intersection over union is 0/0, which is
undefined.</p>
<ul>
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">0</span></code> (default), the consistency for that collection of error sets is set to zero.</p></li>
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">1</span></code>, the consistency for that collection of error sets is set to one.</p></li>
<li><p>If “nan”, the consistency for that collection of error sets is set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nan</span></code>.</p></li>
<li><p>If “drop”, the <code class="xref py py-obj docutils literal notranslate"><span class="pre">consistencies</span></code> array will not include results for that collection,
but the consistency matrix will include <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nans</span></code>.</p></li>
<li><p>If “error”, an empty union will cause a <code class="xref py py-obj docutils literal notranslate"><span class="pre">ZeroDivisionError</span></code>.</p></li>
<li><p>If “warn”, an empty union will print a warning (probably a lot).</p></li>
</ul>
</p></li>
</ul>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>Conceptually, for each repetition, there are two steps to computing a k-fold error consistency
with holdout set:</p>
<blockquote>
<div><ol class="arabic simple">
<li><p>evaluation on standard k-fold (“validation” or “folding”)</p></li>
<li><p>evaluation on holdout set (outside of k-fold) (“testing”)</p></li>
</ol>
</div></blockquote>
<p>There are a lot of overlapping terms and concepts here, so with analogy to deep learning, we
shall refer to step (1) as <em>validation</em> or <em>val</em> and step (2) as <em>testing</em> or <em>test</em>. This will
help keep variable names and function arguments sane and clear. We refer to the <em>entire</em> process
of validation + testing as <em>evaluation</em>. Thus the .evaluate() method with have both validation
and testing steps, in this terminology.</p>
<p>Since validation is very much just standard k-fold, we also thus refer to validation steps as
<em>fold</em> steps. So for example validation or fold scores are the k accuracies on the non-training
partitions of each k-fold repetition (k*repetitions total), but test scores are the
<code class="xref py py-obj docutils literal notranslate"><span class="pre">repititions</span></code> accuracies on the heldout test set.</p>
<dl class="py method">
<dt id="error_consistency.consistency.ErrorConsistencyKFoldInternal.evaluate">
<code class="sig-name descname"><span class="pre">evaluate</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">repetitions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">5</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_errors</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_test_predictions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_preds</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">save_fold_models</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">show_progress</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">True</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">parallel_reps</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">loo_parallel</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">turbo</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">seed</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.consistency.ErrorConsistencyKFoldInternal.evaluate" title="Permalink to this definition">¶</a></dt>
<dd><p>Evaluate the error consistency of the classifier.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>repetitions</strong> (<em>int = 5</em>) – How many times to repeat the k-fold process. Yields <code class="xref py py-obj docutils literal notranslate"><span class="pre">repetitions*(repititions</span> <span class="pre">-</span> <span class="pre">1)/2</span></code>
consistencies if <code class="xref py py-obj docutils literal notranslate"><span class="pre">repetitions</span></code> is greater than 1. Setting repetitions to 1 instead uses
the entire set <code class="xref py py-obj docutils literal notranslate"><span class="pre">X</span></code> for prediction for each fold, thus yield <code class="xref py py-obj docutils literal notranslate"><span class="pre">k*(k-1)/2</span></code> consistencies,
but which are strongly biased toward a value much lower than the true consistency.
Useful for quick checks / fast estimates of upper bounds on the error consistency, but
otherwise not recommended.</p></li>
<li><p><strong>save_test_accs</strong> (<em>bool = True</em>) – If True (default) also compute accuracy scores for each fold on <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_test</span></code> and save them
in <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.scores</span></code>. If False, skip this step. Setting to <code class="xref py py-obj docutils literal notranslate"><span class="pre">False</span></code> is useful when
prediction is expensive and/or you only care about evaulating the error consistency.</p></li>
<li><p><strong>save_test_errors</strong> (<em>bool = False</em>) – If True, save a list of the boolean error arrays (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_i</span> <span class="pre">!=</span> <span class="pre">y_test</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for
all repetitions in <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_errors</span></code>. Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1.
If False (default), <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_errors</span></code> will be <code class="xref py py-obj docutils literal notranslate"><span class="pre">None</span></code>.</p></li>
<li><p><strong>save_test_predictions</strong> (<em>bool = False</em>) – If True, save an array of the predictions <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code> for all repetitions in
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_predictions</span></code>. Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1. If False
(default), <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.test_predictions</span></code> will be <code class="xref py py-obj docutils literal notranslate"><span class="pre">None</span></code>.</p></li>
<li><p><strong>save_fold_accs</strong> (<em>bool = False</em>) – If True, save an array of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k)</span></code> of the predictions on the <em>fold</em> test
set (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_fold_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for all repetitions in <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_accs</span></code>.</p></li>
<li><p><strong>save_fold_preds</strong> (<em>bool = False</em>) – If True, save a NumPy array of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k,</span> <span class="pre">n_samples)</span></code> of the predictions on
the <em>fold</em> test set (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_fold_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for all repetitions in
<code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_predictions</span></code>.</p></li>
<li><p><strong>save_fold_models</strong> (<em>bool = False</em>) – If True, <code class="xref py py-obj docutils literal notranslate"><span class="pre">results.fold_models</span></code> is a NumPy object array of size (repetitions, k) where
each entry (r, i) is the fitted model on repetition <code class="xref py py-obj docutils literal notranslate"><span class="pre">r</span></code> fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>.</p></li>
<li><p><strong>seed</strong> (<em>int = None</em>) – Seed for reproducible results</p></li>
<li><p><strong>show_progress</strong> (<em>bool</em>) – </p></li>
<li><p><strong>parallel_reps</strong> (<em>Union</em><em>[</em><em>bool</em><em>, </em><em>int</em><em>]</em>) – </p></li>
<li><p><strong>loo_parallel</strong> (<em>Union</em><em>[</em><em>bool</em><em>, </em><em>int</em><em>]</em>) – </p></li>
<li><p><strong>turbo</strong> (<em>bool</em>) – </p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p><strong>results</strong> – The <a class="reference internal" href="index.html#error_consistency.containers.ConsistencyResults" title="error_consistency.containers.ConsistencyResults"><code class="xref py py-obj docutils literal notranslate"><span class="pre">error_consistency.containers.ConsistencyResults</span></code></a> object.</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p><a class="reference internal" href="index.html#error_consistency.containers.ConsistencyResults" title="error_consistency.containers.ConsistencyResults">ConsistencyResults</a></p>
</dd>
</dl>
</dd></dl>

</dd></dl>

<dl class="py class">
<dt id="error_consistency.consistency.ErrorConsistencyMonteCarlo">
<em class="property"><span class="pre">class</span> </em><code class="sig-prename descclassname"><span class="pre">error_consistency.consistency.</span></code><code class="sig-name descname"><span class="pre">ErrorConsistencyMonteCarlo</span></code><a class="headerlink" href="#error_consistency.consistency.ErrorConsistencyMonteCarlo" title="Permalink to this definition">¶</a></dt>
<dd><p>Calculate error consistency using repeated random train/test splits.</p>
</dd></dl>

</div>
<span id="document-functional"></span><div class="section" id="module-error_consistency.functional">
<span id="functional-tools"></span><h2>Functional Tools<a class="headerlink" href="#module-error_consistency.functional" title="Permalink to this headline">¶</a></h2>
<dl class="py data">
<dt id="error_consistency.functional.UnionHandling">
<code class="sig-prename descclassname"><span class="pre">error_consistency.functional.</span></code><code class="sig-name descname"><span class="pre">UnionHandling</span></code><a class="headerlink" href="#error_consistency.functional.UnionHandling" title="Permalink to this definition">¶</a></dt>
<dd><p>UnionHandling</p>
<p>alias of Literal[0, 1, nan, drop, warn, error]</p>
</dd></dl>

<dl class="py function">
<dt id="error_consistency.functional.error_consistencies">
<code class="sig-prename descclassname"><span class="pre">error_consistency.functional.</span></code><code class="sig-name descname"><span class="pre">error_consistencies</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">y_preds</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_true</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">empty_unions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">loo_parallel</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">turbo</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">log_progress</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">False</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.functional.error_consistencies" title="Permalink to this definition">¶</a></dt>
<dd><p>Get the error consistency for a list of predictions.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>y_preds</strong> (<em>List</em><em>[</em><em>ndarray</em><em>]</em>) – A list of numpy arrays of predictions, all on the same test set.</p></li>
<li><p><strong>y_true</strong> (<em>ndarray</em>) – The true values against which each of <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_preds</span></code> will be compared.</p></li>
<li><p><strong>sample_dim</strong> (<em>int = 0</em>) – The dimension along which samples are indexed for <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_true</span></code> and each array of <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_preds</span></code>.</p></li>
<li><p><strong>empty_unions</strong> (<em>UnionHandling = 0</em>) – <p>When computing the pairwise consistency or leave-one-out consistency on small or
simple datasets, it can be the case that the union of the error sets is empty (e.g. if no
prediction errors are made). In this case the intersection over union is 0/0, which is
undefined.</p>
<ul>
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">0</span></code> (default), the consistency for that collection of error sets is set to zero.</p></li>
<li><p>If <code class="xref py py-obj docutils literal notranslate"><span class="pre">1</span></code>, the consistency for that collection of error sets is set to one.</p></li>
<li><p>If “nan”, the consistency for that collection of error sets is set to <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nan</span></code>.</p></li>
<li><p>If “drop”, the <code class="xref py py-obj docutils literal notranslate"><span class="pre">consistencies</span></code> array will not include results for that collection,
but the consistency matrix will include <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.nans</span></code>.</p></li>
<li><p>If “error”, an empty union will cause a <code class="xref py py-obj docutils literal notranslate"><span class="pre">ZeroDivisionError</span></code>.</p></li>
<li><p>If “warn”, an empty union will print a warning (probably a lot).</p></li>
</ul>
</p></li>
<li><p><strong>loo_parallel</strong> (<em>bool = False</em>) – If True, use multiprocessing to parallelize the computation of the leave-one-out error
consistencies.</p></li>
<li><p><strong>turbo</strong> (<em>bool = False</em>) – If True, use Numba-accelerated error consistency calculation.</p></li>
<li><p><strong>log_progress</strong> (<em>bool = False</em>) – If True, show a progress bar when computing the leave-one-out error consistencies.</p></li>
</ul>
</dd>
<dt class="field-even">Returns</dt>
<dd class="field-even"><p><ul class="simple">
<li><p><strong>consistencies</strong> (<em>ndarray</em>) – An array of the computed consistency values. Length will depend on <code class="xref py py-obj docutils literal notranslate"><span class="pre">empty_unions</span></code>.</p></li>
<li><p><strong>matrix</strong> (<em>ndarray</em>) – An array of size <code class="docutils literal notranslate"><span class="pre">(N,</span> <span class="pre">N)</span></code> of the pairwise consistency values (IOUs) where
<code class="xref py py-obj docutils literal notranslate"><span class="pre">N</span> <span class="pre">=</span> <span class="pre">len(y_preds)</span></code>, and where entry <code class="docutils literal notranslate"><span class="pre">(i,</span> <span class="pre">j)</span></code> is the pairwise IOU for predictions <code class="docutils literal notranslate"><span class="pre">i</span></code> and
predictions <code class="docutils literal notranslate"><span class="pre">j</span></code>.</p></li>
<li><p><strong>intersection</strong> (<em>ndarray</em>) – The total intersection of all error sets. When nonzero, can be useful for identifying
unpredictable samples.</p></li>
<li><p><strong>union</strong> (<em>ndarray</em>) – The total union of all error sets. Will almost always be non-empty except for trivial
datasets, and thus computing <code class="docutils literal notranslate"><span class="pre">np.sum(intersection)</span> <span class="pre">/</span> <span class="pre">np.sum(union)</span></code> gives something like a
lower bound on the consistencies.</p></li>
<li><p><strong>loo_consistencies</strong> (<em>ndarray</em>) – The IOUs or consistencies computed from applying the union and intesection operations over
all combinations of <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_preds</span></code> of size <code class="xref py py-obj docutils literal notranslate"><span class="pre">len(y_preds)</span> <span class="pre">-</span> <span class="pre">1</span></code>. Sort of a symmetric counterpart
to the default pairwise consistency.</p></li>
</ul>
</p>
</dd>
<dt class="field-odd">Return type</dt>
<dd class="field-odd"><p>Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]</p>
</dd>
</dl>
</dd></dl>

<dl class="py function">
<dt id="error_consistency.functional.get_y_error">
<code class="sig-prename descclassname"><span class="pre">error_consistency.functional.</span></code><code class="sig-name descname"><span class="pre">get_y_error</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">y_pred</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_true</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.functional.get_y_error" title="Permalink to this definition">¶</a></dt>
<dd><p>Returns the</p>
<dl class="field-list simple">
<dt class="field-odd">Returns</dt>
<dd class="field-odd"><p><strong>error_set</strong> – The boolean array with length <code class="xref py py-obj docutils literal notranslate"><span class="pre">n_samples</span></code> where y_pred != y_true. For one-hot or dummy <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code>,
this is still computed such that the length of the returned array is <code class="xref py py-obj docutils literal notranslate"><span class="pre">n_samples</span></code>.</p>
</dd>
<dt class="field-even">Return type</dt>
<dd class="field-even"><p>ndarray</p>
</dd>
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>y_pred</strong> (<em>numpy.ndarray</em>) – </p></li>
<li><p><strong>y_true</strong> (<em>numpy.ndarray</em>) – </p></li>
<li><p><strong>sample_dim</strong> (<em>int</em>) – </p></li>
</ul>
</dd>
</dl>
</dd></dl>

</div>
<span id="document-containers"></span><div class="section" id="module-error_consistency.containers">
<span id="containers"></span><h2>Containers<a class="headerlink" href="#module-error_consistency.containers" title="Permalink to this headline">¶</a></h2>
<dl class="py class">
<dt id="error_consistency.containers.ConsistencyResults">
<em class="property"><span class="pre">class</span> </em><code class="sig-prename descclassname"><span class="pre">error_consistency.containers.</span></code><code class="sig-name descname"><span class="pre">ConsistencyResults</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">consistencies</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">matrix</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">total_consistency</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">leave_one_out_consistency</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">test_errors</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">test_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">test_predictions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fold_accs</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fold_predictions</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fold_models</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.containers.ConsistencyResults" title="Permalink to this definition">¶</a></dt>
<dd><p>Holds results from evaluating error consistency.</p>
<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.consistencies">
<code class="sig-name descname"><span class="pre">consistencies</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.consistencies" title="Permalink to this definition">¶</a></dt>
<dd><p>A flat array of all the pairwise consistencies. Length will be N*(N-1)/2, where for <code class="xref py py-obj docutils literal notranslate"><span class="pre">n_rep</span></code>
reptitions of k-fold, <code class="xref py py-obj docutils literal notranslate"><span class="pre">N</span> <span class="pre">=</span> <span class="pre">n_rep</span> <span class="pre">*</span> <span class="pre">k</span></code> unless the <code class="xref py py-obj docutils literal notranslate"><span class="pre">empty_unions</span></code> method handling was “drop”.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>ndarray</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.matrix">
<code class="sig-name descname"><span class="pre">matrix</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.matrix" title="Permalink to this definition">¶</a></dt>
<dd><p>A NumPy array of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(N,N)</span></code> where <code class="xref py py-obj docutils literal notranslate"><span class="pre">N</span> <span class="pre">=</span> <span class="pre">n_rep</span> <span class="pre">*</span> <span class="pre">k</span></code> for <code class="xref py py-obj docutils literal notranslate"><span class="pre">n_rep</span></code> repetitions of k-fold, and
where <code class="xref py py-obj docutils literal notranslate"><span class="pre">matrix[i,j]</span></code> holds the consistency for pairings <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code> and <code class="xref py py-obj docutils literal notranslate"><span class="pre">j</span></code></p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>ndarray</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.total_consistency">
<code class="sig-name descname"><span class="pre">total_consistency</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.total_consistency" title="Permalink to this definition">¶</a></dt>
<dd><p>Given the <code class="xref py py-obj docutils literal notranslate"><span class="pre">N</span></code> predictions on the test set, where <code class="xref py py-obj docutils literal notranslate"><span class="pre">N</span> <span class="pre">=</span> <span class="pre">n_rep</span> <span class="pre">*</span> <span class="pre">k</span></code> for <code class="xref py py-obj docutils literal notranslate"><span class="pre">n_rep</span></code> repetitions of
k-fold, this is the value of the size of the intersection of all error sets divided
by the size of the union of all those error sets. That is, this is the size of the set of
all samples that were <em>always</em> consistently predicted incorrectly divided by the size of the
set of all samples that had at least one wrong prediction. When the total_consistency is
nonzero, this thus means that there are samples which are <em>always</em> incorrectly predicted
regarded of the training set. This thus be thought of as something like a <em>lower bound</em> on
the consistency estimate, where a non-zero value here is indicative / interesting.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>float</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.leave_one_out_consistency">
<code class="sig-name descname"><span class="pre">leave_one_out_consistency</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.leave_one_out_consistency" title="Permalink to this definition">¶</a></dt>
<dd><p>Given the <code class="xref py py-obj docutils literal notranslate"><span class="pre">N</span></code> predictions on the test set, where <code class="xref py py-obj docutils literal notranslate"><span class="pre">N</span> <span class="pre">=</span> <span class="pre">n_rep</span> <span class="pre">*</span> <span class="pre">k</span></code> for <code class="xref py py-obj docutils literal notranslate"><span class="pre">n_rep</span></code> repetitions of
k-fold, this is the value of the size of the intersection of all error sets <em>excluding one</em>
divided by the size of the union of all those error sets <em>excluding the same one</em>, for each
excluded error set. See README.md for <code class="xref py py-obj docutils literal notranslate"><span class="pre">p-1</span></code> consistency. This is a slightly less punishing
lower-bound that the total_consistency, and is more symmetric with the pairwise consistency.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>float</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.test_errors">
<code class="sig-name descname"><span class="pre">test_errors</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.test_errors" title="Permalink to this definition">¶</a></dt>
<dd><p>A list of the boolean error arrays (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_i</span> <span class="pre">!=</span> <span class="pre">y_test</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for all repetitions.
Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>Optional[List[ndarray]] = None</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.test_accs">
<code class="sig-name descname"><span class="pre">test_accs</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.test_accs" title="Permalink to this definition">¶</a></dt>
<dd><p>An array of the accuracies <code class="xref py py-obj docutils literal notranslate"><span class="pre">np.mean(y_pred_i</span> <span class="pre">==</span> <span class="pre">y_test)</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code> for all repetitions.
Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>Optional[ndarray] = None</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.test_predictions">
<code class="sig-name descname"><span class="pre">test_predictions</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.test_predictions" title="Permalink to this definition">¶</a></dt>
<dd><p>An array of the predictions <code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code> for all repetitions. Total of
<code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>Optional[ndarray] = None</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.fold_accs">
<code class="sig-name descname"><span class="pre">fold_accs</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.fold_accs" title="Permalink to this definition">¶</a></dt>
<dd><p>An array of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k)</span></code> the accuracies (<code class="xref py py-obj docutils literal notranslate"><span class="pre">np.mean(y_pred_fold_i</span> <span class="pre">==</span> <span class="pre">y_fold_i</span></code> for
fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for all repetitions. Total of <code class="xref py py-obj docutils literal notranslate"><span class="pre">k</span> <span class="pre">*</span> <span class="pre">repetitions</span></code> values if k &gt; 1.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>Optional[ndarray] = None</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.fold_predictions">
<code class="sig-name descname"><span class="pre">fold_predictions</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.fold_predictions" title="Permalink to this definition">¶</a></dt>
<dd><p>A NumPy array of shape <code class="xref py py-obj docutils literal notranslate"><span class="pre">(repetitions,</span> <span class="pre">k,</span> <span class="pre">n_samples)</span></code> of the predictions on the <em>fold</em> test
set (<code class="xref py py-obj docutils literal notranslate"><span class="pre">y_pred_fold_i</span></code> for fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>) for all repetitions.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>Optional[ndarray] = None</p>
</dd>
</dl>
</dd></dl>

<dl class="py attribute">
<dt id="error_consistency.containers.ConsistencyResults.fold_models">
<code class="sig-name descname"><span class="pre">fold_models</span></code><a class="headerlink" href="#error_consistency.containers.ConsistencyResults.fold_models" title="Permalink to this definition">¶</a></dt>
<dd><p>A NumPy object array of size (repetitions, k) where each entry (r, i) is the fitted model on
repetition <code class="xref py py-obj docutils literal notranslate"><span class="pre">r</span></code> fold <code class="xref py py-obj docutils literal notranslate"><span class="pre">i</span></code>.</p>
<dl class="field-list simple">
<dt class="field-odd">Type</dt>
<dd class="field-odd"><p>Optional[ndarray[<a class="reference internal" href="index.html#error_consistency.model.Model" title="error_consistency.model.Model">Model</a>]] = None</p>
</dd>
</dl>
</dd></dl>

</dd></dl>

</div>
<span id="document-model"></span><div class="section" id="module-error_consistency.model">
<span id="model-tools"></span><h2>Model Tools<a class="headerlink" href="#module-error_consistency.model" title="Permalink to this headline">¶</a></h2>
<dl class="py class">
<dt id="error_consistency.model.Model">
<em class="property"><span class="pre">class</span> </em><code class="sig-prename descclassname"><span class="pre">error_consistency.model.</span></code><code class="sig-name descname"><span class="pre">Model</span></code><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">model</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">model_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">fit_args_x_y</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">predict_args_x</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">s_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">y_sample_dim</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#error_consistency.model.Model" title="Permalink to this definition">¶</a></dt>
<dd><p>Helper class for making a unified interface to different model types.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><ul class="simple">
<li><p><strong>model</strong> (<em>Intersection</em><em>[</em><em>Callable</em><em>, </em><em>Type</em><em>]</em>) – <p>A <em>class</em> for which instances implement (1) .fit or .train methods and (2) .predict or .test
method, and which takes <code class="xref py py-obj docutils literal notranslate"><span class="pre">model_args</span></code> in its constructor. E.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="kn">from</span> <span class="nn">error_consistency</span> <span class="kn">import</span> <span class="n">ErrorConsistency</span>
<span class="kn">from</span> <span class="nn">sklearn.cluster</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span> <span class="k">as</span> <span class="n">KNN</span>

<span class="n">knn_args</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="n">n_neighbors</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">errcon</span> <span class="o">=</span> <span class="n">ErrorConsistency</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="n">KNN</span><span class="p">,</span> <span class="n">model_args</span><span class="o">=</span><span class="n">knn_args</span><span class="p">)</span>
</pre></div>
</div>
</p></li>
<li><p><strong>model_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time to construct a fresh instance of the model (see
above). Note that the data x and y must NOT be included here.</p></li>
<li><p><strong>fit_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> methods
internally (see notes for <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> above). Note that the data x and y must NOT be included
here.</p></li>
<li><p><strong>fit_args_x_y</strong> (<em>Optional</em><em>[</em><em>Tuple</em><em>[</em><em>str</em><em>, </em><em>str</em><em>]</em><em>] </em><em>= None</em>) – <p>Name of the arguments which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> and target <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> are passed to. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code>.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">train</span></code> method of the instance of
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, and <code class="xref py py-obj docutils literal notranslate"><span class="pre">y</span></code> as its second, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.fit(x,</span> <span class="pre">y,</span> <span class="pre">**model_args)</span></code>.</p>
<p>If a tuple of strings (x_name, y_name), then a dict will be constructed internally by
splatting, e.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">args_dict</span> <span class="o">=</span> <span class="p">{</span><span class="o">**</span><span class="p">{</span><span class="n">x_name</span><span class="p">:</span> <span class="n">x_train</span><span class="p">,</span> <span class="n">y_name</span><span class="p">:</span> <span class="n">y_train</span><span class="p">},</span> <span class="o">**</span><span class="n">model_args</span><span class="p">}</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="o">**</span><span class="n">args_dict</span><span class="p">)</span>
</pre></div>
</div>
</p></li>
<li><p><strong>predict_args</strong> (<em>Optional</em><em>[</em><em>Dict</em><em>[</em><em>str</em><em>, </em><em>Any</em><em>]</em><em>]</em>) – Any arguments that are required each time when calling the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> methods
internally (see notes for <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> above). Note that the data x must NOT be included here.</p></li>
<li><p><strong>predict_args_x</strong> (<em>Optional</em><em>[</em><em>str</em><em>] </em><em>= None</em>) – <p>Name of the argument which data <code class="xref py py-obj docutils literal notranslate"><span class="pre">x</span></code> is passed to during evaluation. This is needed because
different libraries may have different conventions for how they expect predictors and
targets to be passed in to <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> calls.</p>
<p>If None (default), it will be assumed that the <code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code> or <code class="xref py py-obj docutils literal notranslate"><span class="pre">test</span></code> method of the instance
of <code class="xref py py-obj docutils literal notranslate"><span class="pre">model</span></code> takes x as its first positional argument, as in e.g.
<code class="xref py py-obj docutils literal notranslate"><span class="pre">model.predict(x,</span> <span class="pre">**predict_args)</span></code>.</p>
<p>If a string <code class="xref py py-obj docutils literal notranslate"><span class="pre">x_name</span></code>, then a dict will be constructed internally by splatting, e.g.:</p>
<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="n">args_dict</span> <span class="o">=</span> <span class="p">{</span><span class="o">**</span><span class="p">{</span><span class="n">x_name</span><span class="p">:</span> <span class="n">x_train</span><span class="p">},</span> <span class="o">**</span><span class="n">model_args</span><span class="p">}</span>
<span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="o">**</span><span class="n">args_dict</span><span class="p">)</span>
</pre></div>
</div>
</p></li>
<li><p><strong>x_sample_dim</strong> (<em>int = 0</em>) – The axis or dimension along which samples are indexed. Needed for splitting x into
partitions for k-fold.</p></li>
<li><p><strong>y_sample_dim</strong> (<em>int = 0</em>) – The axis or dimension along which samples are indexed. Needed for splitting y into
partitions for k-fold only if the target is e.g. one-hot encoded or dummy-coded.</p></li>
</ul>
</dd>
</dl>
</dd></dl>

</div>
<span id="document-theory"></span><div class="section" id="error-consistency-theory">
<h2>Error Consistency Theory<a class="headerlink" href="#error-consistency-theory" title="Permalink to this headline">¶</a></h2>
<p><strong>NOTE</strong> This section is currently a work in progress.</p>
<div class="section" id="choice-of-groupings">
<h3>Choice of Groupings<a class="headerlink" href="#choice-of-groupings" title="Permalink to this headline">¶</a></h3>
<p>Ultimately the default error consistency metric is somewhat arbitrarily focused on pairwise
estimates of consistency. It would be more accurate to thus call it <em>pairwise</em> error consistency,
and include options for calculating e.g. <em>total</em> error consistency and <span class="math notranslate nohighlight">\(p\)</span>-error consistency for
each partition-size <span class="math notranslate nohighlight">\(k\)</span>.</p>
<p>That is, given <span class="math notranslate nohighlight">\(n = n_{\text{rep}} \cdot k\)</span> predictions <span class="math notranslate nohighlight">\(\{\hat{\mathbf{y}}_i\}_{i=1}^{n}\)</span>, and true
target values <span class="math notranslate nohighlight">\(\mathbf{y}\)</span>, we get the boolean error vectors
<span class="math notranslate nohighlight">\(\mathbf{e}_i = (\hat{\mathbf{y}}_i\ == \mathbf{y})\)</span>. Get all <em>combinations</em> of size 2 of these
error vectors, i.e. the set
<span class="math notranslate nohighlight">\(\mathbf{G}^2 = \left\{G_{ij} = \{\mathbf{e}_i, \mathbf{e}_i\} \;|\; i &lt; j\right\}\)</span>, where
<span class="math notranslate nohighlight">\(|\mathbf{G}^2| = {n\choose 2} = n_2\)</span>. Indexing these combinations with <span class="math notranslate nohighlight">\(k\)</span> in the natural manner,
(so e.g. <span class="math notranslate nohighlight">\(k = 1\)</span> corresponds to <span class="math notranslate nohighlight">\((i, j) = (1, 1)\)</span>, <span class="math notranslate nohighlight">\(k = 2\)</span> corresponds to
<span class="math notranslate nohighlight">\((i, j) = (1, 2)\)</span> and so
on, with always <span class="math notranslate nohighlight">\(i &lt; j\)</span>, we can write <span class="math notranslate nohighlight">\(\mathbf{G}^2 = \left\{G_k \;|\; 1 \le k \le n_2\right\}\)</span>.
We then define the (pairwise) error consistency as the set of <span class="math notranslate nohighlight">\(n_2\)</span> real values:</p>
<div class="math notranslate nohighlight">
\[C^2_k =
\frac{|\mathbf{e}_i \cap \mathbf{e}_j|}{|\mathbf{e}_i \cup \mathbf{e}_j|} =
\frac{\left|\bigcap_{k=1}^{2} G_k \right|}{ \left|\bigcup_{i=1}^{2} G_k \right| }\]</div>
<p>Note the choice of 2 here is largely completely arbitrary, and we could just as well define
<span class="math notranslate nohighlight">\(n\choose3\)</span> consistencies for the set
<span class="math notranslate nohighlight">\(\mathbf{G}_3 = \left\{ G_{i_1,i_2,i_3} = \{\mathbf{e}_{i_1}, \mathbf{e}_{i_2}, \mathbf{e}_{i_3}\} \;|\; i_1 &lt; i_2 &lt; i_j \right\}\)</span>,
with <span class="math notranslate nohighlight">\(k\)</span> as before:</p>
<div class="math notranslate nohighlight">
\[C^3_k =
\frac
{|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \cap \mathbf{e}_{i_3}|}
{|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \cup \mathbf{e}_{i_3}|}
=
\frac{\left|\bigcap_{k=1}^{3} G^3_k \right|}{ \left|\bigcup_{i=1}^{3} G^3_k \right| }\]</div>
<p>Likewise, we can define the <span class="math notranslate nohighlight">\(p\)</span>-wise error consistency for <span class="math notranslate nohighlight">\(\mathbf{G}_p\)</span>:</p>
<div class="math notranslate nohighlight">
\[C^p_k =
\frac
{|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \dots \cap \mathbf{e}_{i_p}|}
{|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \dots \cup \mathbf{e}_{i_p}|}
=
\frac{\left|\bigcap_{k=1}^{p} G^p_k \right|}{ \left|\bigcup_{i=1}^{p} G^p_k \right| },
\quad p \in \{2, 3, \dots, n\}\]</div>
<p>For convenience, note we can also define <span class="math notranslate nohighlight">\(C_k^1 = 1\)</span>, which is consistent with the above definition.</p>
<p>Generally, we will be most interested in summary statistics of the set <span class="math notranslate nohighlight">\(\mathbf{C}^p = \{C^p_k\}\)</span>, such
as <span class="math notranslate nohighlight">\(\bar{\mathbf{C}^p}\)</span> and <span class="math notranslate nohighlight">\(\text{var}({\mathbf{C}^p})\)</span>.</p>
<p><strong>Note:</strong> With particularly poor and/or inconsistent classifiers, it can quite easily happen for
some values of <span class="math notranslate nohighlight">\(k\)</span> that <span class="math notranslate nohighlight">\(\bigcup_{i=1}^{p} G^p_k = \varnothing\)</span>, which would leave the above equations
undefined. In practice, we just drop such combinations and consider only non-empty unions.</p>
</div>
<div class="section" id="why-p-2">
<h3>Why p=2<a class="headerlink" href="#why-p-2" title="Permalink to this headline">¶</a></h3>
<p>While the “true” error consistency we would likely wish to describe as something like the average
over all valid <span class="math notranslate nohighlight">\(p\)</span>, this is obviously computationally intractable even for very small <span class="math notranslate nohighlight">\(n\)</span> and <span class="math notranslate nohighlight">\(p\)</span>.
However, quite clearly:</p>
<div class="math notranslate nohighlight">
\[1 \ge C_k^1 \ge \max(\mathbf{C}^2) \ge \dots \ge \max(\mathbf{C^{p-1}}) \ge \max(\mathbf{C}^p)\]</div>
<p>with equality holding very rarely (when most predictions are identical). In fact, we can make a much
stronger statement, namely:</p>
<div class="math notranslate nohighlight">
\[C_k^{p+1} \le \max(\mathbf{C}^p) \;\;\forall p\]</div>
<p>since the inclusion of more sets in the intersection can only decrease the numerator, and increase
the denominator. Thus, <span class="math notranslate nohighlight">\(\max(\mathbf{C}^2)\)</span> provides an upper bound on the consistency. In fact,
since <span class="math notranslate nohighlight">\(\mathbf{G}^p \subset \mathbf{G}^{p+1}\)</span> for all <span class="math notranslate nohighlight">\(p\)</span>, we necessarily for all <span class="math notranslate nohighlight">\(k\)</span>, then we can
state something even stronger:</p>
<div class="math notranslate nohighlight">
\[C^{p+1}_k =
\frac
{|\mathbf{e}_{i_1} \cap \mathbf{e}_{i_2} \dots \cap \mathbf{e}_{i_{p+1}}|}
{|\mathbf{e}_{i_1} \cup \mathbf{e}_{i_2} \dots \cup \mathbf{e}_{i_{p+1}}|}
\le
\frac
{|\mathbf{e}_{i} \cap \mathbf{e}_{j}|}
{|\mathbf{e}_{i} \cup \mathbf{e}_{j}|}
=
C^{p}_{ij} \text{for all } i, j \in \{i_1, \dots, i_p+1\} \text{ where } i &lt; j\]</div>
<p>Equality will hold only for combinations of error sets where the error sets are identical.
However, if you do the counting, sFor each <span class="math notranslate nohighlight">\(k\)</span> and each <span class="math notranslate nohighlight">\(p\)</span>, there are in fact
<span class="math notranslate nohighlight">\({p\choose 2} = p(p-1)/2\)</span> such unique consistencies <span class="math notranslate nohighlight">\(C^{p}_{ij}\)</span> which are larger than <span class="math notranslate nohighlight">\(C^{p+1}_k\)</span>.
E.g. consider <span class="math notranslate nohighlight">\(p=3, 4, 5\)</span>, and that <span class="math notranslate nohighlight">\(i, j\)</span> and <span class="math notranslate nohighlight">\(k\)</span>, <span class="math notranslate nohighlight">\(\dots\)</span> stand in for any sequence of ascending numbers,
e.g. <span class="math notranslate nohighlight">\((i,j,k,l,m) = (1,2,5,7,9)\)</span>. Then:</p>
<div class="math notranslate nohighlight">
\[\begin{split}\begin{align}
C^3_{k'} &amp;= C^3_{ijk} \le (C^2_{ij},C^2_{ik}),\;(C^2_{jk})\\
C^4_{k'} &amp;= C^4_{ijkl} \le (C^2_{ij},C^2_{ik}, C^2_{il}),\; (C^2_{jk},C^2_{jl}),\; (C^2_{kl})\\
C^5_{k'} &amp;= C^5_{ijklm} \le (C^2_{ij}, C^2_{ik}, C^2_{il}), C^2_{im}),\; (C^2_{jk},C^2_{jl},
C^2_{jm}),\; (C^2_{kl}, C^2_{km}), (C^2_{lm})
\end{align}\end{split}\]</div>
<p>Clearly three are <span class="math notranslate nohighlight">\(1 + 2 + \dots (p-1) = p(p-1)/2\)</span> values each time, since we always require our
indices to be less than each other. However, since <span class="math notranslate nohighlight">\(C^p_{k'}\)</span> is less than <em>all</em> these values, it
is also less than the <em>smallest</em> of those values, and the <em>average</em> of those values. For different
values of <span class="math notranslate nohighlight">\(k\)</span>, the smallest member of <span class="math notranslate nohighlight">\(\mathbf{C}^2\)</span> may be the same. But there are at
most <span class="math notranslate nohighlight">\(n\choose2\)</span> unique values in <span class="math notranslate nohighlight">\(\mathbf{C}^2\)</span>.</p>
<div class="math notranslate nohighlight">
\[C^{p+1}_k
\le
\min(\mathbf{C}_k^2)
\le
\frac{1}{p\choose2}\sum_i^{p\choose2}C^{p}_{k_i}
= \text{mean}(\mathbf{C}_k^2) \text{ for some }
\mathbf{C}_k^2 \subset \mathbf{C}^2\]</div>
<p>Thus</p>
<div class="math notranslate nohighlight">
\[\sum_k^{n\choose{p+1}}C^{p+1}_k
\le
\sum_k^{n\choose{p+1}}\min(\mathbf{C}_k^2)
\le
\sum_k^{n\choose{p+1}}C^{p}_{j_k} \text{for some } j_k\]</div>
<p>In fact it should be clear</p>
<p>## Total Error Consistency</p>
</div>
</div>
</div>
</div>
<div class="section" id="indices-and-tables">
<h1>Indices and tables<a class="headerlink" href="#indices-and-tables" title="Permalink to this headline">¶</a></h1>
<ul class="simple">
<li><p><a class="reference internal" href="genindex.html"><span class="std std-ref">Index</span></a></p></li>
<li><p><a class="reference internal" href="py-modindex.html"><span class="std std-ref">Module Index</span></a></p></li>
<li><p><a class="reference internal" href="search.html"><span class="std std-ref">Search Page</span></a></p></li>
</ul>
</div>


           </div>
           
          </div>
          <footer>

  <hr/>

  <div role="contentinfo">
    <p>
        &#169; Copyright 2021, Derek Berger, Jacob Levman.

    </p>
  </div>
    
    
    
    Built with <a href="https://www.sphinx-doc.org/">Sphinx</a> using a
    
    <a href="https://github.com/readthedocs/sphinx_rtd_theme">theme</a>
    
    provided by <a href="https://readthedocs.org">Read the Docs</a>. 

</footer>
        </div>
      </div>

    </section>

  </div>
  

  <script type="text/javascript">
      jQuery(function () {
          SphinxRtdTheme.Navigation.enable(true);
      });
  </script>

  
  
    
   

</body>
</html>
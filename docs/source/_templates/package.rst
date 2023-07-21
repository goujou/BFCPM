{{ name | escape | underline }}

PPP

.. currentmodule:: {{ fullname }}


.. automodule:: {{ fullname }}
   :members:
   :inherited-members:
   :show-inheritance:
 
   {% block modules %}
   {% if modules %}
   .. rubric:: Modules

   .. autosummary::
      :toctree:
      :recursive:
      :template: module.rst

      {% for item in modules %}
        {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}



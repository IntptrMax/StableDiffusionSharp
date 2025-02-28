using StableDiffusionSharp;
using System.Diagnostics;

namespace StableDiffusionDemo_Winform
{
	public partial class FormMain : Form
	{
		string modelPath = string.Empty;
		StableDiffusion sd;

		public FormMain()
		{
			InitializeComponent();
		}

		private void FormMain_Load(object sender, EventArgs e)
		{
			sd = new StableDiffusion();
			sd.StepProgress += Sd_StepProgress;
		}

		private void Sd_StepProgress(object? sender, StableDiffusion.StepEventArgs e)
		{
			Label_State.Text = $"Processing {e.CurrentStep}/{e.TotalSteps}";
		}

		private void Button_ModelScan_Click(object sender, EventArgs e)
		{
			FileDialog fileDialog = new OpenFileDialog();
			fileDialog.Filter = "Model files (*.safetensors)|*.safetensors";
			if (fileDialog.ShowDialog() == DialogResult.OK)
			{
				TextBox_ModelPath.Text = fileDialog.FileName;
				modelPath = fileDialog.FileName;
			}
		}

		private void Button_ModelLoad_Click(object sender, EventArgs e)
		{
			if (File.Exists(modelPath))
			{
				Task.Run(() =>
				{
					base.Invoke((Action)delegate
					{
						Button_ModelLoad.Enabled = false;
					});
					sd.LoadModel(modelPath);
					base.Invoke((Action)delegate
					{
						Button_ModelLoad.Enabled = true;
						Button_Generate.Enabled = true;
						Label_State.Text = "Model loaded.";	
					});
				});
			}
			else
			{
				MessageBox.Show("Please select a valid model file.");
			}
		}

		private void Button_Generate_Click(object sender, EventArgs e)
		{
			string prompt = TextBox_Prompt.Text;
			string nprompt = TextBox_NPrompt.Text;
			int step = (int)NumericUpDown_Step.Value;
			float cfg = (float)NumericUpDown_CFG.Value;
			ulong seed = 0;
			int width = (int)NumericUpDown_Width.Value;
			int height = (int)NumericUpDown_Height.Value;

			Task.Run(() =>
			{
				Stopwatch stopwatch = Stopwatch.StartNew();
				base.Invoke((Action)delegate
				{
					Button_ModelLoad.Enabled = false;
					Button_Generate.Enabled = false;
					Label_State.Text = "Generating...";
				});
				ImageMagick.MagickImage image = sd.TextToImage(prompt, nprompt, width, height, step, seed, cfg);
				MemoryStream memoryStream = new MemoryStream();
				image.Write(memoryStream, ImageMagick.MagickFormat.Jpg);
				base.Invoke((Action)delegate
				{
					PictureBox_Output.Image = Image.FromStream(memoryStream);
					Button_ModelLoad.Enabled = true;
					Button_Generate.Enabled = true;
					Label_State.Text = $"Done. It takes {stopwatch.Elapsed.TotalSeconds.ToString("f2")} s";
				});
				GC.Collect();
			});



		}
	}
}
